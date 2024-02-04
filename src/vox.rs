use bevy::{
    asset::{AssetLoader, AsyncReadExt},
    prelude::*,
};
use dot_vox::{DotVoxData, Model, SceneNode};

use crate::chunk::{ChunkHandle, ChunkMaterial, ChunkOctree, TiledChunker};

#[derive(Component, Clone, Copy)]
pub struct VoxChunk;

#[derive(Component, Clone, Copy)]
pub struct VoxScene;

#[derive(Asset, TypePath)]
pub struct Vox(DotVoxData);

#[derive(Default)]
pub struct VoxLoader;
impl Plugin for VoxLoader {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_asset_loader::<VoxLoader>()
            .init_asset::<Vox>()
            .add_systems(Startup, test_load)
            .add_systems(Update, on_vox_asset_event);
    }
}

fn test_load(asset_server: Res<AssetServer>, mut commands: Commands) {
    // TODO: decide what to do with this Vox oject. it is already loaded by VoxLoader
    commands.spawn((
        // asset_server.load::<Vox>("./castle_pro.vox"),
        asset_server.load::<Vox>("./castle.vox"),
        GlobalTransform::default(),
    ));
}

impl AssetLoader for VoxLoader {
    type Asset = Vox;
    type Settings = ();
    type Error = anyhow::Error;

    fn load<'a>(
        &'a self,
        reader: &'a mut bevy::asset::io::Reader,
        _settings: &'a Self::Settings,
        // NOTE: could use this to load DotVoxData directly into Handle<Image> and stuff,
        // but it will also need to spawn chunks, so not doing that here.
        _load_context: &'a mut bevy::asset::LoadContext,
    ) -> bevy::utils::BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async {
            let mut data = vec![];
            reader.read_to_end(&mut data).await?;
            let data = dot_vox::load_bytes(&data).map_err(anyhow::Error::msg)?;
            Ok(Vox(data))
        })
    }

    fn extensions(&self) -> &[&str] {
        &["vox"]
    }
}

fn on_vox_asset_event(
    mut events: EventReader<AssetEvent<Vox>>,
    vox_assets: Res<Assets<Vox>>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut chunk_materials: ResMut<Assets<ChunkMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
) {
    for event in events.read() {
        let AssetEvent::LoadedWithDependencies { id } = event else {
            continue;
        };
        let data = vox_assets.get(*id).expect("event said it is loaded");

        load_vox(
            &mut commands,
            &mut images,
            &mut chunk_materials,
            &mut meshes,
            &mut std_materials,
            &data.0,
        );
    }
}

struct VoxModel {
    model: Model,
    translation: IVec3,
    rotation: Quat,
    scale: Vec3,
}
struct VoxParser<'a> {
    data: &'a DotVoxData,
    models: Vec<VoxModel>,
}
impl VoxParser<'_> {
    fn parse(&mut self, scene_index: usize, translation: IVec3, rotation: Quat, scale: Vec3) {
        let scene = &self.data.scenes[scene_index];
        match scene {
            SceneNode::Transform { frames, child, .. } => {
                assert_eq!(frames.len(), 1, "unimplemented");
                let frame = &frames[0];
                let translation = translation
                    + frame
                        .position()
                        .map(|p| IVec3::new(p.x as _, p.z as _, p.y as _))
                        .unwrap_or_default();
                let (rotation, scale) = frame
                    .orientation()
                    .map(|o| o.to_quat_scale())
                    .map(|(q, f)| {
                        (
                            Quat::from_xyzw(q[0], q[2], -q[1], q[3]),
                            Vec3::new(f[0], f[2], f[1]),
                        )
                    })
                    .unwrap_or_default();

                self.parse(*child as _, translation, rotation, scale);
            }
            SceneNode::Group { children, .. } => {
                children.iter().for_each(|child| {
                    self.parse(*child as _, translation, rotation, scale);
                });
            }
            SceneNode::Shape {
                models: shape_models,
                ..
            } => {
                assert_eq!(shape_models.len(), 1, "unimplemented");
                let model = &self.data.models[shape_models[0].model_id as usize];

                // let mut offset = Vec3::new(
                //     if model.size.x % 2 == 0 { 0.0 } else { 0.5 },
                //     if model.size.z % 2 == 0 { 0.0 } else { 0.5 },
                //     if model.size.y % 2 == 0 { 0.0 } else { -0.5 },
                // );
                // offset = rotation.mul_vec3(offset);
                // let center = rotation
                //     * (Vec3::new(model.size.x as _, model.size.y as _, model.size.z as _)
                //         / 2.0);
                // let translation = translation - center * scale + offset;

                let vmodel = VoxModel {
                    model: Model {
                        voxels: model
                            .voxels
                            .iter()
                            .map(|v| dot_vox::Voxel {
                                x: v.x,
                                y: v.z,
                                z: v.y,
                                i: v.i + 1,
                            })
                            .collect(),
                        size: dot_vox::Size {
                            x: model.size.x,
                            y: model.size.z,
                            z: model.size.y,
                        },
                    },
                    translation,
                    rotation,
                    scale,
                };
                self.models.push(vmodel);
            }
        }
    }

    fn get_materials(&self) -> Vec<Vec4> {
        let materials = [Vec4::splat(0.0)]
            .into_iter()
            .chain(
                self.data
                    .palette
                    .iter()
                    .map(|c| Vec4::new(c.r as f32, c.g as f32, c.b as f32, c.a as f32)),
            )
            .map(|c| c / 256.0)
            .take(256) // idk why this buffer has 1 extra material :/
            .collect::<Vec<_>>();
        materials
    }

    fn tiled_chunker(&self, chunk_side: usize) -> TiledChunker {
        let mut tc = TiledChunker::with_chunk_size(chunk_side);

        for model in self.models.iter() {
            let chunk_pos = model.translation;
            let pivot = Vec3::new(
                (model.model.size.x / 2) as _,
                (model.model.size.y / 2) as _,
                (model.model.size.z / 2) as _,
            );
            // TODO: inserting each voxel using tc.set_voxel is gonna be slow as each voxel will require a hashmap.get
            for voxel in &model.model.voxels {
                let mut voxel_pos = IVec3::new(voxel.x as _, voxel.y as _, voxel.z as _);

                // if model.scale.x.is_sign_negative() {
                //     voxel_pos.x = model.model.size.x as i32 - voxel.x as i32 - 1;
                // }
                // if model.scale.y.is_sign_negative() {
                //     voxel_pos.x = model.model.size.y as i32 - voxel.y as i32 - 1;
                // }
                // if model.scale.z.is_sign_negative() {
                //     voxel_pos.x = model.model.size.z as i32 - voxel.z as i32 - 1;
                // }

                // https://github.com/jpaver/opengametools/blob/a45c00f7c4ff2e7f750099cba7fcc299e0d08096/src/ogt_vox.h#L120
                let voxel_pos =
                    Vec3::new(voxel_pos.x as _, voxel_pos.y as _, voxel_pos.z as _) - pivot;
                let voxel_pos = model.rotation.mul_vec3(voxel_pos);
                let voxel_pos = IVec3::new(
                    voxel_pos.x.round() as _,
                    voxel_pos.y.round() as _,
                    voxel_pos.z.round() as _,
                );

                let pivot = IVec3::new(
                    (model.model.size.x / 2) as _,
                    (model.model.size.y / 2) as _,
                    (model.model.size.z / 2) as _,
                );
                tc.set_voxel(chunk_pos + voxel_pos, voxel.i)
                    .expect("invalid index");
            }
        }

        // let keys = tc.chunks.iter().map(|(k, v)| k).collect::<Vec<_>>();
        // dbg!(keys);

        tc
    }

    // very similar to self.tiled_chunker()
    /// chunk_size in log2
    fn svo(
        &self,
        chunk_height: u32,
        entity: Entity,
        images: &mut Assets<Image>,
        meshes: &mut Assets<Mesh>,
    ) -> ChunkOctree {
        let material_handle = images.add(ChunkHandle::material_image(self.get_materials()));
        let cube_handle = meshes.add(Mesh::from(shape::Cube { size: 2.0 }));

        let mut svo = ChunkOctree::new(
            chunk_height + 5,
            chunk_height,
            cube_handle,
            material_handle,
            entity,
        );

        for model in self.models.iter() {
            let chunk_pos = model.translation;
            let pivot = Vec3::new(
                (model.model.size.x / 2) as _,
                (model.model.size.y / 2) as _,
                (model.model.size.z / 2) as _,
            );

            for voxel in &model.model.voxels {
                let voxel_pos = IVec3::new(voxel.x as _, voxel.y as _, voxel.z as _);

                let voxel_pos =
                    Vec3::new(voxel_pos.x as _, voxel_pos.y as _, voxel_pos.z as _) - pivot;
                let voxel_pos = model.rotation.mul_vec3(voxel_pos);
                let voxel_pos = IVec3::new(
                    voxel_pos.x.round() as _,
                    voxel_pos.y.round() as _,
                    voxel_pos.z.round() as _,
                );

                svo.set_voxel(chunk_pos + voxel_pos, voxel.i)
                    .expect("invalid index");
            }
        }
        svo.update_mip();

        svo
    }
}

fn load_vox(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    chunk_materials: &mut Assets<ChunkMaterial>,
    meshes: &mut Assets<Mesh>,
    std_materials: &mut Assets<StandardMaterial>,
    data: &DotVoxData,
) {
    let vox_scene_entity = commands
        .spawn((
            Name::new("VoxScene"),
            VoxScene,
            GlobalTransform::default(),
            Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            VisibilityBundle::default(),
        ))
        .id();

    let mut parser = VoxParser {
        data,
        models: vec![],
    };
    parser.parse(0, IVec3::default(), Quat::default(), Vec3::ZERO);

    // let mut svo = parser.svo(7, vox_scene_entity, images, meshes);
    // svo.spawn_chunks(commands, images, chunk_materials);
    // commands.entity(vox_scene_entity).insert(svo);

    let material_handle = images.add(ChunkHandle::material_image(parser.get_materials()));
    let cube_handle = meshes.add(Mesh::from(shape::Cube { size: 2.0 }));
    let tc = parser.tiled_chunker(128);
    tc.spawn_chunks(
        vox_scene_entity,
        commands,
        images,
        chunk_materials,
        1.0 / 16.0,
        &material_handle,
        &cube_handle,
        IVec3::ZERO,
    );
}
