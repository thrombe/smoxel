use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    window::{CursorGrabMode, PrimaryWindow},
};

use crate::ControlsState;

#[derive(Component)]
pub struct PlayerEntity;

pub struct PlayerPlugin;
impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, update.run_if(in_state(ControlsState::Player)));
    }
}

fn setup(mut commands: Commands, mut _meshes: ResMut<Assets<Mesh>>) {
    let transform = Transform::from_xyz(5.0, 5.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        Camera3dBundle {
            transform,
            ..default()
        },
        PlayerEntity,
    ));
}

fn update(
    mut player: Query<(&mut Transform, &mut PlayerEntity)>,
    keys: Res<Input<KeyCode>>,
    mouse_keys: Res<Input<MouseButton>>,
    mut mouse_events: EventReader<MouseMotion>,
    // time: Res<Time>,
    mut window: Query<&mut Window, With<PrimaryWindow>>,
) {
    let mut wind = window.single_mut();
    if keys.just_pressed(KeyCode::Escape) {
        wind.cursor.grab_mode = match wind.cursor.grab_mode {
            CursorGrabMode::Locked => {
                wind.cursor.visible = true;
                CursorGrabMode::None
            }
            _ => {
                wind.cursor.visible = false;
                CursorGrabMode::Locked
            }
        };
        let pos = Vec2::new(
            wind.resolution.width() / 2.0,
            wind.resolution.height() / 2.0,
        );
        wind.set_cursor_position(Some(pos));
        mouse_events.clear();
    }

    let (mut transform, _) = player.single_mut();
    if wind.cursor.grab_mode != CursorGrabMode::Locked {
        let pos = Vec2::new(
            wind.resolution.width() / 2.0,
            wind.resolution.height() / 2.0,
        );
        if mouse_keys.just_pressed(MouseButton::Left) {
            wind.cursor.visible = false;
            wind.cursor.grab_mode = CursorGrabMode::Confined;
            wind.set_cursor_position(Some(pos));
            mouse_events.clear();
        } else if mouse_keys.just_released(MouseButton::Left) {
            wind.cursor.visible = true;
            wind.cursor.grab_mode = CursorGrabMode::None;
        } else if mouse_keys.pressed(MouseButton::Left) {
            wind.set_cursor_position(Some(pos));
        } else {
            return;
        }
    }

    let mut mouse_delta = Vec2::ZERO;
    for e in mouse_events.read() {
        mouse_delta += e.delta;
    }
    mouse_delta *= 0.002;

    {
        let mut rotation = transform.rotation;
        let yrotation = Quat::from_axis_angle(Vec3::Y, -mouse_delta.x);
        let xrotation = Quat::from_axis_angle(transform.local_x(), -mouse_delta.y);

        rotation = xrotation * rotation;
        let is_upside_down = rotation.mul_vec3(Vec3::Y).y < 0.0;
        if is_upside_down {
            rotation = transform.rotation;
        }

        rotation = yrotation * rotation;
        transform.rotation = rotation;
    }

    let speed = 0.2;
    let forward = transform.forward();
    let right = transform.local_x();
    let mut translation = Vec3::ZERO;

    if keys.pressed(KeyCode::W) {
        translation += forward * speed;
    }
    if keys.pressed(KeyCode::S) {
        translation -= forward * speed;
    }
    if keys.pressed(KeyCode::A) {
        translation -= right * speed;
    }
    if keys.pressed(KeyCode::D) {
        translation += right * speed;
    }
    if keys.pressed(KeyCode::ShiftLeft) {
        translation *= 5.0;
    }
    if keys.pressed(KeyCode::ControlLeft) {
        translation /= 5.0;
    }
    transform.translation += translation;
}

pub mod spectator {
    use super::*;

    #[derive(Component)]
    pub struct SpectatorEntity;

    pub struct Spectator;
    impl Plugin for Spectator {
        fn build(&self, app: &mut App) {
            app.add_systems(Startup, setup)
                .add_systems(Update, switch_mode)
                .add_systems(Update, update.run_if(in_state(ControlsState::Spectator)));
        }
    }

    fn switch_mode(
        mut keys: ResMut<Input<KeyCode>>,
        mut spectator: Query<(&mut Camera, &mut SpectatorEntity)>,
        curr_state: Res<State<ControlsState>>,
        mut state: ResMut<NextState<ControlsState>>,
    ) {
        if !keys.just_pressed(KeyCode::Key1) {
            return;
        }

        let (mut camera, _) = spectator.single_mut();

        if *curr_state.get() == ControlsState::Spectator {
            state.set(ControlsState::Player);
            camera.is_active = false;
        } else {
            state.set(ControlsState::Spectator);
            camera.is_active = true;
        }

        // dosen't fix the bug T-T
        // keys.press(KeyCode::Escape);
    }

    fn setup(mut commands: Commands, mut _meshes: ResMut<Assets<Mesh>>) {
        let transform = Transform::from_xyz(5.0, 5.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y);
        commands.spawn((
            Camera3dBundle {
                transform,
                camera: Camera {
                    order: 1,
                    is_active: false,
                    ..Default::default()
                },
                ..default()
            },
            SpectatorEntity,
        ));
    }

    fn update(
        mut spectator: Query<(&mut Transform, &mut SpectatorEntity)>,
        keys: Res<Input<KeyCode>>,
        mouse_keys: Res<Input<MouseButton>>,
        mut mouse_events: EventReader<MouseMotion>,
        // time: Res<Time>,
        mut window: Query<&mut Window, With<PrimaryWindow>>,
    ) {
        let mut wind = window.single_mut();
        if keys.just_pressed(KeyCode::Escape) {
            wind.cursor.grab_mode = match wind.cursor.grab_mode {
                CursorGrabMode::Locked => {
                    wind.cursor.visible = true;
                    CursorGrabMode::None
                }
                _ => {
                    wind.cursor.visible = false;
                    CursorGrabMode::Locked
                }
            };
            let pos = Vec2::new(
                wind.resolution.width() / 2.0,
                wind.resolution.height() / 2.0,
            );
            wind.set_cursor_position(Some(pos));
            mouse_events.clear();
        }

        let (mut transform, _) = spectator.single_mut();
        if wind.cursor.grab_mode != CursorGrabMode::Locked {
            let pos = Vec2::new(
                wind.resolution.width() / 2.0,
                wind.resolution.height() / 2.0,
            );
            if mouse_keys.just_pressed(MouseButton::Left) {
                wind.cursor.visible = false;
                wind.cursor.grab_mode = CursorGrabMode::Confined;
                wind.set_cursor_position(Some(pos));
                mouse_events.clear();
            } else if mouse_keys.just_released(MouseButton::Left) {
                wind.cursor.visible = true;
                wind.cursor.grab_mode = CursorGrabMode::None;
            } else if mouse_keys.pressed(MouseButton::Left) {
                wind.set_cursor_position(Some(pos));
            } else {
                return;
            }
        }

        let mut mouse_delta = Vec2::ZERO;
        for e in mouse_events.read() {
            mouse_delta += e.delta;
        }
        mouse_delta *= 0.002;

        {
            let mut rotation = transform.rotation;
            let yrotation = Quat::from_axis_angle(Vec3::Y, -mouse_delta.x);
            let xrotation = Quat::from_axis_angle(transform.local_x(), -mouse_delta.y);

            rotation = xrotation * rotation;
            let is_upside_down = rotation.mul_vec3(Vec3::Y).y < 0.0;
            if is_upside_down {
                rotation = transform.rotation;
            }

            rotation = yrotation * rotation;
            transform.rotation = rotation;
        }

        let speed = 0.2;
        let forward = transform.forward();
        let right = transform.local_x();
        let mut translation = Vec3::ZERO;

        if keys.pressed(KeyCode::W) {
            translation += forward * speed;
        }
        if keys.pressed(KeyCode::S) {
            translation -= forward * speed;
        }
        if keys.pressed(KeyCode::A) {
            translation -= right * speed;
        }
        if keys.pressed(KeyCode::D) {
            translation += right * speed;
        }
        if keys.pressed(KeyCode::ShiftLeft) {
            translation *= 5.0;
        }
        if keys.pressed(KeyCode::ControlLeft) {
            translation /= 5.0;
        }
        transform.translation += translation;
    }
}
