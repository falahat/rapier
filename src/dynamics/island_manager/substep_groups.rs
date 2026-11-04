//! Substep solve groups: the awake set partitioned by effective `additional_solver_iterations`
//! and `additional_pgs_iterations` (max over each awake connected component), grouped by count
//! into contiguous ranges. Applied (reorder, re-stamp, epoch bump) only when not already grouped.
//! With `parallel-groups`, each touching set of jointed bodies that nothing else is solved
//! against is a group of its own, solved beside the others.

use super::IslandManager;
use crate::data::union_find::UnionFind;
use crate::dynamics::{
    ImpulseJointSet, MultibodyJointSet, RigidBodyHandle, RigidBodySet, SoftBodySet,
};
use crate::geometry::{ColliderSet, ContactPair, NarrowPhase};
use alloc::vec::Vec;
use core::ops::Range;

/// A contiguous range of the awake island's bodies sharing one substep count. Ranges are ordered
/// by decreasing `extra_iters`, so kinematic bodies (never merged, assigned to the highest-cadence
/// group they touch) are integrated before any lower-cadence group solves against them.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub(crate) struct SolveGroup {
    /// Range into the awake island's `bodies` (== solver-body slot range).
    pub body_range: Range<usize>,
    /// Extra substeps for this group, on top of
    /// `IntegrationParameters::num_solver_iterations`.
    pub extra_iters: u32,
    /// Extra internal PGS iterations per substep for this group, on top of
    /// `IntegrationParameters::num_internal_pgs_iterations`.
    pub extra_pgs: u32,
    /// Whether the group is one touching set of jointed bodies alone: it shares no body,
    /// contact or joint with any other group, so it can be solved beside them.
    pub separate: bool,
}

/// The partition key of a body: its extra substeps in the high half, its extra PGS iterations
/// in the low half, so the groups sort by substep cadence first.
#[inline]
fn key(extra_substeps: u32, extra_pgs: u32) -> u32 {
    (extra_substeps.min(0xFFFF) << 16) | extra_pgs.min(0xFFFF)
}

/// A body that is a jointed body's link, and one coupled to a kinematic or soft body:
/// marks a touching set takes from its bodies (see [`SubstepGroupsWorkspace::marks`]).
const JOINTED: u8 = 1;
const SHARED: u8 = 2;

/// The element-wise maximum of two keys (a component takes the largest of each count).
#[inline]
fn max_key(a: u32, b: u32) -> u32 {
    key((a >> 16).max(b >> 16), (a & 0xFFFF).max(b & 0xFFFF))
}

/// Workspace state for [`IslandManager::update_substep_groups`], kept to reuse
/// allocations across steps.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub(crate) struct SubstepGroupsWorkspace {
    uf: UnionFind,
    /// Effective key per awake body slot (component max of each count, see `key`).
    keys: Vec<u32>,
    /// Touching-set rank per awake body slot: its component's root slot plus one for a
    /// separate set (see [`SolveGroup::separate`]), 0 for every other body.
    ranks: Vec<u32>,
    /// [`JOINTED`] and [`SHARED`] marks per awake body slot, then OR-ed per component root.
    marks: Vec<u8>,
    sets: Vec<u8>,
    /// The reorder's permutation, and a scratch buffer to apply it to the keys and ranks.
    order: Vec<u32>,
    scratch: Vec<u32>,
    /// Reorder workspace for the awake `bodies` vec.
    workspace: Vec<RigidBodyHandle>,
}

impl IslandManager {
    /// Recomputes the awake set's substep solve-groups, reordering the awake island's body list so
    /// groups are contiguous (descending extra count). Must run after `update_islands` (last
    /// mutator of the awake body list) and before anything consumes body order or stamps solver-body indices. `any_extra` = caller's OR of `additional_solver_iterations > 0`; when `false` this is one branch + `Vec::clear` (empty groups = one implicit whole-set group).
    pub(crate) fn update_substep_groups(
        &mut self,
        any_extra: bool,
        bodies: &mut RigidBodySet,
        narrow_phase: &NarrowPhase,
        colliders: &ColliderSet,
        impulse_joints: &ImpulseJointSet,
        multibody_joints: &MultibodyJointSet,
        soft_bodies: &SoftBodySet,
    ) {
        self.solve_groups.clear();
        // Touching sets are worth setting apart only where they are solved side by side, and
        // only with two jointed bodies to set apart.
        let split =
            cfg!(feature = "parallel-groups") && multibody_joints.multibodies().nth(1).is_some();
        if !any_extra && !split {
            return;
        }
        let Some(awake_id) = self.awake_island else {
            return;
        };
        let num_bodies = self.islands[awake_id].bodies.len();

        let ws = &mut self.substep_groups_workspace;
        ws.uf.reset(num_bodies);
        ws.marks.clear();
        ws.marks.resize(num_bodies, 0);

        // An edge merges components only between awake *dynamic* bodies: fixed/kinematic bodies
        // are not solver DOFs and must not glue unrelated components (a kinematic platform would
        // merge an elevated assembly with the default world); sleeping neighbors are read-only walls.
        let slot = |handle: RigidBodyHandle| -> Option<u32> {
            let rb = bodies.get(handle)?;
            (rb.is_dynamic() && !rb.is_sleeping() && rb.ids.active_island_id == awake_id as u32)
                .then_some(rb.ids.active_set_id)
        };

        // Contact edges from the narrow-phase's current pairs; solver-side filters are ignored as
        // under-merging would split coupled bodies' cadences. A soft body's colliders hang on its
        // cluster proxies, so a pair touching a soft surface couples it like any other pair.
        for pair in narrow_phase.contact_pairs() {
            if !pair.has_any_active_contact() {
                continue;
            }
            if let (Some(h1), Some(h2)) = pair_bodies(pair, colliders)
                && let (Some(s1), Some(s2)) = (slot(h1), slot(h2))
            {
                ws.uf.union(s1, s2);
            }
        }

        // Impulse-joint edges. All joints are considered (not just this step's
        // active selection, which isn't computed yet at this point): a joint
        // between two awake dynamic bodies couples them regardless.
        for (_, joint) in impulse_joints.iter() {
            if let (Some(s1), Some(s2)) = (slot(joint.body1), slot(joint.body2)) {
                ws.uf.union(s1, s2);
            }
        }

        // Multibody-joint edges: each joint links a body to its parent link's
        // body. Chaining parent unions connects every dynamic link of an
        // articulation even when its root is fixed.
        for (_, link_id, multibody, link) in multibody_joints.iter() {
            if let Some(parent) = multibody.link(link.parent_internal_id)
                && let (Some(s1), Some(s2)) = (
                    slot(parent.rigid_body_handle()),
                    slot(link.rigid_body_handle()),
                )
            {
                ws.uf.union(s1, s2);
            }
            let _ = link_id;
        }

        // Soft-body attachments: a soft body (its root) is coupled with the bodies its particles
        // are attached to, and its cluster proxies with each other (one substep cadence per
        // soft body).
        for (_, sb) in soft_bodies.iter() {
            let Some(root) = slot(sb.root_body()) else {
                continue;
            };
            ws.marks[root as usize] |= SHARED;
            for attachment in sb.particle_attachments() {
                if let Some(s) = slot(attachment.body) {
                    ws.uf.union(root, s);
                }
            }
            for (_, cluster) in sb.live_clusters() {
                if let Some(s) = slot(cluster.proxy()) {
                    ws.uf.union(root, s);
                }
            }
        }

        // Effective counts per body = max of each over its component; non-dynamic awake bodies
        // stay singletons with key 0, then are lifted to the highest-cadence group they touch.
        ws.keys.clear();
        ws.keys.resize(num_bodies, 0);
        for i in 0..num_bodies {
            let handle = self.islands[awake_id].bodies[i];
            let rb = &bodies[handle];
            let body_key = key(
                rb.additional_solver_iterations() as u32,
                rb.additional_pgs_iterations() as u32,
            );
            if body_key > 0 {
                let root = ws.uf.find(i as u32) as usize;
                ws.keys[root] = max_key(ws.keys[root], body_key);
            }
        }
        for i in 0..num_bodies {
            let root = ws.uf.find(i as u32) as usize;
            ws.keys[i] = ws.keys[root];
        }

        // Lift kinematic bodies to the max key among the dynamic bodies they touch. Contacts are
        // the only solver coupling a kinematic body has; joints follow the same rule for consistency.
        let kinematic_slot = |handle: RigidBodyHandle| -> Option<u32> {
            let rb = bodies.get(handle)?;
            (!rb.is_dynamic()
                && rb.is_dynamic_or_kinematic()
                && !rb.is_sleeping()
                && rb.ids.active_island_id == awake_id as u32)
                .then_some(rb.ids.active_set_id)
        };
        let lift = |ws: &mut SubstepGroupsWorkspace, h1, h2| {
            if let (Some(k), Some(d)) = (kinematic_slot(h1), slot(h2)) {
                ws.keys[k as usize] = ws.keys[k as usize].max(ws.keys[d as usize]);
                ws.marks[d as usize] |= SHARED;
            }
        };
        for pair in narrow_phase.contact_pairs() {
            if !pair.has_any_active_contact() {
                continue;
            }
            if let (Some(h1), Some(h2)) = pair_bodies(pair, colliders) {
                lift(ws, h1, h2);
                lift(ws, h2, h1);
            }
        }
        for (_, joint) in impulse_joints.iter() {
            lift(ws, joint.body1, joint.body2);
            lift(ws, joint.body2, joint.body1);
        }

        // A touching set that holds a jointed body, and no kinematic or soft body, is a group of
        // its own: nothing else is solved against it, so it can be solved beside the rest. Its
        // bodies take its root slot plus one as their rank, every other body 0, and the groups
        // follow the keys, then the ranks.
        ws.ranks.clear();
        ws.ranks.resize(num_bodies, 0);
        if split {
            for multibody in multibody_joints.multibodies() {
                if let Some(s) = (multibody.links()).find_map(|link| slot(link.rigid_body_handle()))
                {
                    ws.marks[s as usize] |= JOINTED;
                }
            }
            ws.sets.clear();
            ws.sets.resize(num_bodies, 0);
            for i in 0..num_bodies {
                let root = ws.uf.find(i as u32) as usize;
                ws.sets[root] |= ws.marks[i];
            }
            for i in 0..num_bodies {
                let root = ws.uf.find(i as u32);
                if ws.sets[root as usize] == JOINTED {
                    ws.ranks[i] = root + 1;
                }
            }
        }
        let place =
            |ws: &SubstepGroupsWorkspace, i: usize| (core::cmp::Reverse(ws.keys[i]), ws.ranks[i]);

        // Fast path: the list is already grouped (keys non-increasing along the vec, ranks
        // non-decreasing within a key). Steady-state elevated scenes take this branch every
        // step; only derive the ranges, no reorder, no epoch bump.
        if (1..num_bodies).all(|i| place(ws, i - 1) <= place(ws, i)) {
            push_group_ranges(&mut self.solve_groups, &ws.keys, &ws.ranks);
            return;
        }

        // Reorder: a stable sort by key, descending, then by touching set.
        ws.order.clear();
        ws.order.extend(0..num_bodies as u32);
        (ws.order).sort_by_key(|&i| {
            (
                core::cmp::Reverse(ws.keys[i as usize]),
                ws.ranks[i as usize],
            )
        });
        let island_bodies = &mut self.islands[awake_id].bodies;
        ws.workspace.clear();
        (ws.workspace).extend(ws.order.iter().map(|&i| island_bodies[i as usize]));
        core::mem::swap(island_bodies, &mut ws.workspace);
        for list in [&mut ws.keys, &mut ws.ranks] {
            ws.scratch.clear();
            (ws.scratch).extend(ws.order.iter().map(|&i| list[i as usize]));
            core::mem::swap(list, &mut ws.scratch);
        }

        // Re-stamp the invariant `active_set_id == index in bodies`, and invalidate every
        // ordering-derived cache.
        for (i, handle) in self.islands[awake_id].bodies.iter().enumerate() {
            bodies.index_mut_internal(*handle).ids.active_set_id = i as u32;
        }
        push_group_ranges(&mut self.solve_groups, &ws.keys, &ws.ranks);
        self.bump_active_set_epoch();
    }
}

/// The two bodies a touching pair couples: its first manifold's, or the colliders' parents
/// for a pair of two soft surfaces.
fn pair_bodies(
    pair: &ContactPair,
    colliders: &ColliderSet,
) -> (Option<RigidBodyHandle>, Option<RigidBodyHandle>) {
    match pair.manifolds().first() {
        Some(manifold) => (manifold.data.rigid_body1, manifold.data.rigid_body2),
        None => (
            colliders.get(pair.collider1).and_then(|c| c.parent()),
            colliders.get(pair.collider2).and_then(|c| c.parent()),
        ),
    }
}

/// Derives contiguous group ranges from a descending-sorted key sequence and the bodies'
/// touching-set ranks: a group per run of one key and one rank.
fn push_group_ranges(groups: &mut Vec<SolveGroup>, keys: &[u32], ranks: &[u32]) {
    let mut start = 0;
    for i in 1..=keys.len() {
        if i == keys.len() || keys[i] != keys[start] || ranks[i] != ranks[start] {
            groups.push(SolveGroup {
                body_range: start..i,
                extra_iters: keys[start] >> 16,
                extra_pgs: keys[start] & 0xFFFF,
                separate: ranks[start] > 0,
            });
            start = i;
        }
    }
}
