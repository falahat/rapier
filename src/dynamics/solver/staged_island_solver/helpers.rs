//! Staged-solver entry points on [`ContactConstraintsSet`] and
//! [`VelocitySolver`]: generic (multibody) constraint placement and the
//! multibody-only halves of init and position integration.

use crate::dynamics::solver::SolverVel;
use crate::dynamics::solver::VelocitySolver;
use crate::dynamics::solver::contact_constraint::{
    ContactConstraintsSet, GenericContactConstraint, GenericContactConstraintBuilder,
};
use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::dynamics::solver::solver_contact_graph::SolverContactGraph;
#[cfg(feature = "dim3")]
use crate::dynamics::solver::velocity_solver::GyroParams;
use crate::dynamics::{IntegrationParameters, MultibodyJointSet, RigidBodyHandle, RigidBodySet};
use crate::math::{DVector, Real};
use core::ops::Range;
use parry::math::SIMD_WIDTH;

use super::GroupLayout;

impl ContactConstraintsSet {
    /// Places the generic (multibody) contact constraints, each taking its jacobian entries in
    /// turn, for the staged solver's workers to generate.
    pub(super) fn place_generic_constraints(
        &mut self,
        bodies: &RigidBodySet,
        multibody_joints: &MultibodyJointSet,
        graph: &SolverContactGraph,
        store: &ManifoldStore,
    ) {
        let generic = graph.generic();
        self.generic_velocity_constraints_builder
            .resize(generic.len(), GenericContactConstraintBuilder::invalid());
        (self.generic_velocity_constraints)
            .resize(generic.len(), GenericContactConstraint::invalid());
        let mut jacobian_id = 0;
        for (r, constraint) in generic.iter().zip(&mut self.generic_velocity_constraints) {
            constraint.place(
                *r,
                store.get(*r),
                bodies,
                multibody_joints,
                &mut jacobian_id,
            );
        }
        crate::dynamics::solver::grow_jacobians(&mut self.generic_jacobians, jacobian_id);
    }
}

impl VelocitySolver {
    /// Initializes the solver buffers and gives each multibody its dofs, group by group; the
    /// workers initialize the rigid bodies and the multibodies themselves (see
    /// [`Self::init_multibody`]) in their first stages.
    pub(super) fn init_solver_buffers_and_multibodies(
        &mut self,
        groups: &mut [GroupLayout],
        island_bodies: &[RigidBodyHandle],
        num_awake_soft_mass_points: usize,
        multibodies: &mut MultibodyJointSet,
    ) {
        self.multibody_roots.clear();
        self.solver_bodies.clear();

        // The island's rigid bodies, then the awake soft bodies' particles.
        let aligned_solver_bodies_len =
            (island_bodies.len() + num_awake_soft_mass_points).div_ceil(SIMD_WIDTH) * SIMD_WIDTH;
        self.solver_bodies.resize(aligned_solver_bodies_len);

        self.solver_vels_increment.clear();
        self.solver_vels_increment
            .resize(aligned_solver_bodies_len, SolverVel::zero());

        // Reset every step so multibody/padding slots (which the body-copy stage
        // skips) read as gyro-disabled.
        #[cfg(feature = "dim3")]
        {
            self.solver_gyro.clear();
            self.solver_gyro
                .resize(aligned_solver_bodies_len, GyroParams::default());
        }

        // Assign solver ids to multibodies, and collect the relevant roots, group by group: a
        // multibody's links share a group, so each group's roots and DOFs are contiguous.
        let mut multibody_solver_id = 0;
        for group in groups.iter_mut() {
            let first_root = self.multibody_roots.len();
            let first_dof = multibody_solver_id as usize;
            for handle in &island_bodies[group.bodies.clone()] {
                if let Some(link) = multibodies.rigid_body_link(*handle).copied() {
                    let multibody = multibodies
                        .get_multibody_mut_internal(link.multibody)
                        .unwrap();

                    if link.id == 0 || link.id == 1 && !multibody.root_is_dynamic {
                        multibody.solver_id = multibody_solver_id;
                        multibody_solver_id += multibody.ndofs() as u32;
                        self.multibody_roots.push(link);
                    }
                }
            }
            group.multibody_roots = first_root..self.multibody_roots.len();
            group.multibody_dofs = first_dof..multibody_solver_id as usize;
        }

        self.generic_solver_vels_increment = DVector::zeros(multibody_solver_id as usize);
        self.generic_solver_vels = DVector::zeros(multibody_solver_id as usize);
    }

    /// Initializes the multibody of `multibody_roots[root]` at the substep `dt` of its links'
    /// group: its links' velocities, mass matrix and accelerations, and its solver velocities.
    pub(super) fn init_multibody(
        &mut self,
        root: usize,
        dt: Real,
        bodies: &mut RigidBodySet,
        multibodies: &mut MultibodyJointSet,
    ) {
        let multibody = multibodies
            .get_multibody_mut_internal(self.multibody_roots[root].multibody)
            .unwrap();
        multibody.update_velocities(bodies);
        multibody.update_mass_matrix(dt, bodies);
        multibody.update_acceleration(dt, bodies);

        let (first, ndofs) = (multibody.solver_id as usize, multibody.ndofs());
        (self.generic_solver_vels_increment.rows_mut(first, ndofs)).axpy(
            dt,
            &multibody.accelerations,
            0.0,
        );
        (self.generic_solver_vels.rows_mut(first, ndofs)).copy_from(&multibody.velocities);
    }

    /// The multibody part of `integrate_positions` for one group's multibodies (`roots`, a range
    /// of `multibody_roots`), used by the staged solver (regular solver bodies are integrated in
    /// parallel by the workers).
    pub(super) fn integrate_multibody_positions(
        &mut self,
        params: &IntegrationParameters,
        roots: Range<usize>,
        is_last_substep: bool,
        bodies: &mut RigidBodySet,
        multibodies: &mut MultibodyJointSet,
    ) {
        for link in &self.multibody_roots[roots] {
            let multibody = multibodies
                .get_multibody_mut_internal(link.multibody)
                .unwrap();
            let solver_vels = self
                .generic_solver_vels
                .rows(multibody.solver_id as usize, multibody.ndofs());
            multibody.velocities.copy_from(&solver_vels);
            multibody.integrate(params.dt);
            multibody.forward_kinematics(bodies, false);
            multibody.update_rigid_bodies_internal(bodies, !is_last_substep, true, false);

            if !is_last_substep {
                multibody.update_velocities(bodies);
                multibody.update_mass_matrix(params.dt, bodies);
                multibody.update_acceleration(params.dt, bodies);

                let mut solver_vels_incr = self
                    .generic_solver_vels_increment
                    .rows_mut(multibody.solver_id as usize, multibody.ndofs());
                solver_vels_incr.axpy(params.dt, &multibody.accelerations, 0.0);
            }
        }
    }
}
