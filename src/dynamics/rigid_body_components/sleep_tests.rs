//! Sleep uses motion of the actual local shape bounds, independent of body origin.

use super::*;
use crate::dynamics::{RigidBodyBuilder, RigidBodyHandle};
use crate::geometry::{ColliderBuilder, SharedShape};
use crate::num::FloatConst;
use crate::pipeline::PhysicsWorld;

fn cuboid(center: Vector, half_extents: Vector) -> (PhysicsWorld, RigidBodyHandle) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let body = world.insert_body(RigidBodyBuilder::dynamic());
    world.insert_collider(
        ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).translation(center),
        Some(body),
    );
    world.step();
    (world, body)
}

#[test]
fn off_center_sleep_displacement_is_frame_invariant() {
    let center = Vector::new(3.0, -2.0, 1.0);
    let half = Vector::new(0.02, 0.03, 1.0);
    let (centered, a) = cuboid(Vector::ZERO, half);
    let (off_center, b) = cuboid(center, half);
    let previous = Pose::from_translation(center);
    let current = Pose::new(center, Vector::Z * 0.0005);
    let mut shifted_current = current;
    shifted_current.translation -= current.rotation * center;

    let centered_drift = centered.bodies[a]
        .mass_properties()
        .max_point_displacement(&previous, &current);
    let shifted_drift = off_center.bodies[b]
        .mass_properties()
        .max_point_displacement(&Pose::IDENTITY, &shifted_current);
    approx::assert_relative_eq!(
        centered_drift,
        shifted_drift,
        epsilon = Real::EPSILON * 32.0
    );
    assert!(shifted_drift < 2.0 * 0.001 / 60.0);
    // The former origin-plus-sphere metric rejects this same physical movement.
    assert!(
        crate::geometry::relative_pose_drift(
            &Pose::IDENTITY,
            &shifted_current,
            off_center.bodies[b].mass_properties().max_extent(),
        ) > 2.0 * 0.001 / 60.0
    );
}

#[test]
fn long_axis_rotation_can_sleep_but_transverse_motion_cannot() {
    let (world, handle) = cuboid(Vector::ZERO, Vector::new(0.02, 0.03, 1.0));
    let mprops = world.bodies[handle].mass_properties();
    let mut axial = RigidBodyActivation::active();
    axial.normalized_linear_threshold = 0.001;
    axial.angular_threshold = 0.001;
    axial.time_until_sleep = 1.0;
    let mut transverse = axial;
    let dt = 1.0 / 60.0;
    for step in 1..=61 {
        let angle = step as Real * 0.0005;
        axial.update_energy(
            RigidBodyType::Dynamic,
            1.0,
            0.0,
            0.03 * 0.03,
            mprops,
            &Pose::new(Vector::ZERO, Vector::Z * angle),
            dt,
        );
        transverse.update_energy(
            RigidBodyType::Dynamic,
            1.0,
            0.0,
            0.03 * 0.03,
            mprops,
            &Pose::new(Vector::ZERO, Vector::X * angle),
            dt,
        );
    }
    assert!(axial.is_eligible_for_sleep());
    assert_eq!(transverse.time_since_can_sleep, 0.0);
}

#[test]
fn corner_displacement_conservatively_bounds_interior_points() {
    let center = Vector::new(3.0, -2.0, 1.0);
    let half = Vector::new(0.02, 0.03, 1.0);
    let (world, handle) = cuboid(center, half);
    let mprops = world.bodies[handle].mass_properties();
    let previous = Pose::new(Vector::new(0.3, -0.7, 2.0), Vector::new(0.2, -0.3, 0.4));
    let current = Pose::new(
        Vector::new(0.31, -0.69, 2.02),
        Vector::new(0.21, -0.31, 0.42),
    );
    let bound = mprops.max_point_displacement(&previous, &current);
    let mut observed: Real = 0.0;
    for x in -2..=2 {
        for y in -2..=2 {
            for z in -2..=2 {
                let point = center + half * Vector::new(x as Real, y as Real, z as Real) / 2.0;
                let distance =
                    (current.transform_point(point) - previous.transform_point(point)).length();
                assert!(distance <= bound + Real::EPSILON * 64.0);
                observed = observed.max(distance);
            }
        }
    }
    // The independent world-point calculation includes all corners, so the bound is tight.
    approx::assert_relative_eq!(bound, observed, epsilon = Real::EPSILON * 64.0);
}

#[test]
fn collider_edits_refresh_sleep_bounds_in_the_existing_change_path() {
    let center = Vector::new(3.0, -2.0, 1.0);
    let half = Vector::new(0.02, 0.03, 1.0);
    let (mut world, body) = cuboid(center, half);
    let collider = world.bodies[body].colliders()[0];
    let initial = Aabb::from_half_extents(center, half);
    assert_eq!(
        world.bodies[body].mass_properties().local_aabb,
        Some(initial)
    );
    approx::assert_relative_eq!(
        world.bodies[body].mass_properties().max_extent(),
        half.length()
    );

    let other = world.insert_collider(
        ColliderBuilder::ball(0.25).translation(Vector::new(-2.0, 0.0, 0.0)),
        Some(body),
    );
    world.step();
    assert_eq!(
        world.bodies[body].mass_properties().local_aabb,
        Some(Aabb::new(
            Vector::new(-2.25, -2.03, -0.25),
            Vector::new(3.02, 0.25, 2.0)
        ))
    );
    world.colliders[other].set_enabled(false);
    world.step();
    assert_eq!(
        world.bodies[body].mass_properties().local_aabb,
        Some(initial)
    );
    assert!(world.remove_collider(other).is_some());

    world.colliders[collider].set_shape(SharedShape::cuboid(0.04, 0.05, 2.0));
    let shifted = center + Vector::X * 2.0;
    world.colliders[collider]
        .set_position_wrt_parent(Pose::new(shifted, Vector::Z * Real::FRAC_PI_2()));
    world.step();
    let bounds = world.bodies[body].mass_properties().local_aabb.unwrap();
    approx::assert_relative_eq!(
        bounds.mins,
        shifted - Vector::new(0.05, 0.04, 2.0),
        epsilon = Real::EPSILON * 32.0
    );
    approx::assert_relative_eq!(
        bounds.maxs,
        shifted + Vector::new(0.05, 0.04, 2.0),
        epsilon = Real::EPSILON * 32.0
    );
    approx::assert_relative_eq!(
        world.bodies[body].mass_properties().max_extent(),
        Vector::new(0.04, 0.05, 2.0).length()
    );

    world.colliders[collider].set_enabled(false);
    world.step();
    assert_eq!(world.bodies[body].mass_properties().local_aabb, None);
    assert_eq!(world.bodies[body].mass_properties().max_extent(), 0.0);
    world.colliders[collider].set_enabled(true);
    world.step();
    assert_eq!(
        world.bodies[body].mass_properties().local_aabb,
        Some(bounds)
    );
    assert!(world.remove_collider(collider).is_some());
    world.step();
    assert_eq!(world.bodies[body].mass_properties().local_aabb, None);
}

#[cfg(feature = "serde-serialize")]
#[test]
fn sleep_geometry_and_timer_continue_across_restore() {
    let (mut world, body) = cuboid(Vector::new(3.0, -2.0, 1.0), Vector::new(0.02, 0.03, 1.0));
    for _ in 0..7 {
        world.step();
    }
    let properties = world.bodies[body].mass_properties();
    let mut missing = serde_json::to_value(properties).unwrap();
    assert!(
        missing
            .as_object_mut()
            .unwrap()
            .remove("local_aabb")
            .is_some()
    );
    let error = serde_json::from_value::<RigidBodyMassProps>(missing).unwrap_err();
    assert!(
        error.to_string().contains("missing field `local_aabb`"),
        "{error}"
    );
    let empty = RigidBodyMassProps::default();
    let explicit_none = serde_json::to_value(&empty).unwrap();
    assert!(explicit_none["local_aabb"].is_null());
    assert_eq!(
        serde_json::from_value::<RigidBodyMassProps>(explicit_none).unwrap(),
        empty
    );
    let bytes = bincode::serialize(&world).unwrap();
    let mut restored: PhysicsWorld = bincode::deserialize(&bytes).unwrap();
    assert_eq!(
        world.bodies[body].mass_properties(),
        restored.bodies[body].mass_properties()
    );
    assert_eq!(
        world.bodies[body].activation(),
        restored.bodies[body].activation()
    );
    for _ in 0..61 {
        world.step();
        restored.step();
        assert_eq!(
            world.bodies[body].mass_properties(),
            restored.bodies[body].mass_properties()
        );
        assert_eq!(
            world.bodies[body].activation(),
            restored.bodies[body].activation()
        );
        assert_eq!(
            world.bodies[body].is_sleeping(),
            restored.bodies[body].is_sleeping()
        );
    }
    assert!(world.bodies[body].is_sleeping());
}
