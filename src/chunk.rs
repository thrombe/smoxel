use std::ops::AddAssign;

use bevy::{
    prelude::*,
    render::{
        extract_component::ExtractComponent,
        render_resource::{AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat},
    },
    tasks::Task,
    utils::HashMap,
};

use block_mesh::{MergeVoxel, Voxel, VoxelVisibility};

use crate::{player::PlayerEntity, vox, AppState};

pub struct VoxelPlugin;
impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Startup,
            (
                setup_voxel_plugin,
                // test_voxels,
            ),
        )
        // .add_systems(OnEnter(AppState::Playing), test_spawn)
        // .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .add_systems(
            Update,
            (
                || {},
                // spawn_chunk_tasks,
                // resolve_chunk_tasks,
                // test_dynamic_mip,
            ),
        )
        .add_event::<ChunkOctreeModified>();
    }
}

fn test_voxels(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut chunk_materials: ResMut<Assets<ChunkMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let scene_pos = IVec3::new(0, 0, 0);
    let scene_entity = commands
        .spawn((
            Name::new("test voxels"),
            GlobalTransform::default(),
            Transform::from_translation(scene_pos.as_vec3()),
            VisibilityBundle::default(),
        ))
        .id();

    let size = 16.0;
    let side = 128usize;
    let voxel_size = size / side as f32;
    let lim = 512;
    let k = 0.09;
    let perlin = noise::Perlin::new(234342);

    let mut tc = TiledChunker::with_chunk_size(side);
    for z in 0..lim {
        for y in 0..lim {
            for x in 0..lim {
                let pos = IVec3::new(x, y, z);

                // let val = perlin.get([x as f64 * k, y as f64 * k, z as f64 * k]);
                let val = rand::random::<f32>();
                if val < 0.001 {
                    tc.set_voxel(pos, rand::random()).expect("should not fail");
                }
            }
        }
        dbg!(z);
    }
    let materials = std::iter::once(Vec4::splat(0.0))
        .chain((1..256).map(|_| {
            Vec4::new(
                rand::random::<f32>() * 0.5 + 0.2,
                rand::random::<f32>() * 0.5 + 0.2,
                rand::random::<f32>() * 0.5 + 0.2,
                1.0,
            )
        }))
        .collect();

    let material_handle = images.add(ChunkHandle::material_image(materials));
    let cube_handle = meshes.add(Mesh::from(shape::Cube { size: 2.0 }));

    tc.spawn_chunks(
        scene_entity,
        &mut commands,
        &mut images,
        &mut chunk_materials,
        voxel_size,
        &material_handle,
        &cube_handle,
        scene_pos,
    );
}

fn test_dynamic_mip(
    mut commands: Commands,
    mut svo: Query<(&mut ChunkOctree, &GlobalTransform)>,
    mut player: Query<(&GlobalTransform, &PlayerEntity)>,
    mut images: ResMut<Assets<Image>>,
    mut chunk_materials: ResMut<Assets<ChunkMaterial>>,
) {
    let (player, _) = player.single();
    for (mut svo, pos) in svo.iter_mut() {
        svo.update_chunks(
            player.translation(),
            IVec3::ZERO,
            &mut commands,
            &mut images,
            &mut chunk_materials,
        );
    }
}

#[derive(Resource)]
pub struct VoxelWorld {
    noise: noise::Perlin,
}

pub struct ChunkFetchInfo<'a> {
    pos: Vec3,
    size: f32,
    chunk_entity: &'a mut Entity,
}

#[derive(Component, Event)]
pub struct ChunkOctreeModified;

#[derive(Component, Clone, Debug)]
pub struct ChunkOctree {
    pub root: ChunkOctreeNode,
    pub height: u32,       // 2.pow(height) number of voxels on edges
    pub chunk_height: u32, // in log2 (-size, size)

    pub voxel_physical_size: f32,
    pub cube_handle: Handle<Mesh>,
    pub materials: Handle<Image>,
    pub entity: Entity,
}
impl ChunkOctree {
    pub fn new(
        height: u32,
        chunk_height: u32,
        cube_handle: Handle<Mesh>,
        materials: Handle<Image>,
        entity: Entity,
    ) -> Self {
        let voxel_physical_size = 1.0 / 16.0;
        Self {
            entity,
            height,
            chunk_height,
            root: ChunkOctreeNode::new_node(chunk_height),
            voxel_physical_size,
            cube_handle,
            materials,
        }
    }
    pub fn set_voxel(&mut self, pos: IVec3, voxel: u8) -> Option<u8> {
        if (IVec3::splat(2i32.pow(self.height - 1)) - pos.abs()).max_element() < 0 {
            return None;
        }
        Some(
            self.root
                .set_voxel(pos, voxel, self.height, self.chunk_height),
        )
    }

    pub fn bit_world(&self) -> BitWorld {
        let mut bw = BitWorld::new();

        self.root.insert_to_bit_world(&mut bw, IVec3::ZERO);

        bw
    }

    pub fn update_mip(&mut self) {
        self.root.update_mip(self.chunk_height);
    }
    pub fn spawn_chunks(
        &mut self,
        commands: &mut Commands,
        images: &mut Assets<Image>,
        chunk_materials: &mut Assets<ChunkMaterial>,
    ) {
        self.root.spawn_chunks(
            self.entity,
            commands,
            images,
            chunk_materials,
            self.voxel_physical_size,
            &self.materials,
            &self.cube_handle,
            IVec3::ZERO,
            self.height,
        );
    }
    fn update_chunks(
        &mut self,
        player: Vec3,
        chunk_pos: IVec3,

        commands: &mut Commands,
        images: &mut Assets<Image>,
        chunk_materials: &mut Assets<ChunkMaterial>,
    ) {
        self.root.set_mip_visibility(commands, Visibility::Hidden);
        self.root.update_chunks(
            player,
            chunk_pos,
            self.height,
            self.chunk_height,
            self.entity,
            commands,
            images,
            chunk_materials,
            self.voxel_physical_size,
            &self.materials,
            &self.cube_handle,
        )
    }
}
#[derive(Clone, Debug)]
pub enum ChunkOctreeNode {
    Node {
        chunks: Box<[Option<Self>; 8]>,
        mip: Chunk,
    },
    Chunk {
        voxels: Chunk,
    },
}
impl ChunkOctreeNode {
    fn new_node(chunk_height: u32) -> Self {
        let byte = ByteChunk::new_with_size(2usize.pow(chunk_height));
        let mip1 = byte.mip();
        let mip2 = mip1.mip();
        let mip = Chunk {
            byte,
            mip1,
            mip2,
            entity: None,
        };
        Self::Node {
            chunks: Default::default(),
            mip,
        }
    }
    fn new_leaf(chunk_height: u32) -> Self {
        let byte = ByteChunk::new_with_size(2usize.pow(chunk_height));
        let mip1 = byte.mip();
        let mip2 = mip1.mip();
        let chunk = Chunk {
            byte,
            mip1,
            mip2,
            entity: None,
        };
        Self::Chunk { voxels: chunk }
    }
    fn mip(&self) -> &Chunk {
        match self {
            Self::Node { mip, .. } => mip,
            Self::Chunk { voxels } => voxels,
        }
    }
    fn mip_mut(&mut self) -> &mut Chunk {
        match self {
            Self::Node { mip, .. } => mip,
            Self::Chunk { voxels } => voxels,
        }
    }

    pub fn insert_to_bit_world(&self, bw: &mut BitWorld, chunk_pos: IVec3) {
        match self {
            ChunkOctreeNode::Node { chunks, mip } => {
                chunks
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| c.as_ref().map(|c| (i, c)))
                    .for_each(|(i, c)| {
                        let mut ipos = chunk_pos * 2;
                        let mut offset = IVec3::ONE;
                        if i & 0b001 == 0 {
                            offset.x = 0;
                        }
                        if i & 0b010 == 0 {
                            offset.y = 0;
                        }
                        if i & 0b100 == 0 {
                            offset.z = 0;
                        }
                        ipos += offset;

                        c.insert_to_bit_world(bw, ipos);
                    });
            }
            ChunkOctreeNode::Chunk { voxels } => {
                bw.insert(&voxels.mip1, chunk_pos.as_uvec3());
            }
        }
    }

    fn despawn_rec(&mut self, commands: &mut Commands) {
        match self {
            Self::Node { chunks, mip } => {
                if let Some(e) = mip.entity.take() {
                    commands.entity(e).despawn();
                } else {
                    chunks
                        .iter_mut()
                        .flatten()
                        .for_each(|c| c.despawn_rec(commands));
                }
            }
            Self::Chunk { voxels } => {
                let _ = voxels.entity.take().map(|e| commands.entity(e).despawn());
            }
        }
    }
    fn set_mip_visibility(&mut self, commands: &mut Commands, vis: Visibility) {
        match self {
            Self::Node { chunks, mip } => {
                if let Some(e) = mip.entity.as_ref() {
                    commands.entity(*e).insert(vis);
                }
            }
            Self::Chunk { voxels } => {
                let _ = voxels
                    .entity
                    .as_ref()
                    .map(|e| commands.entity(*e).insert(vis).id());
            }
        }
    }

    // https://www.desmos.com/calculator/fh2pi1fzfd
    #[allow(clippy::too_many_arguments)]
    fn update_chunks(
        &mut self,
        player: Vec3,
        chunk_pos: IVec3,
        height: u32,
        chunk_height: u32,

        svo_entity: Entity,
        commands: &mut Commands,
        images: &mut Assets<Image>,
        chunk_materials: &mut Assets<ChunkMaterial>,
        voxel_physical_size: f32,
        materials: &Handle<Image>,
        cube_mesh: &Handle<Mesh>,
    ) {
        let size = 2i32.pow(height - 2);
        match self {
            Self::Node { chunks, mip } => {
                if mip.entity.is_none() {
                    mip.spawn(
                        svo_entity,
                        commands,
                        images,
                        chunk_materials,
                        voxel_physical_size * 2i32.pow(height - chunk_height) as f32,
                        materials,
                        cube_mesh,
                        chunk_pos.as_vec3() * voxel_physical_size,
                    );
                }
                let mut ipos = chunk_pos;
                let pos = ipos.as_vec3() * voxel_physical_size;
                let dir = (pos - player).normalize() * voxel_physical_size;
                let close = pos - size as f32 * 2.0 * dir;
                let far = pos + size as f32 * 2.0 * dir;
                let close_dist = (close - player).length().abs();
                let far_dist = (far - player).length().abs();
                let center_dist = (pos - player).length().abs();
                let dist = center_dist.min(close_dist);

                let r = 128.0;
                let mult = 2u32.pow(height - chunk_height + 1) as f32;
                let mut despawn_radius = r * 6.0 * mult;
                let mut spawn_radius = r * 2.0 * mult;
                despawn_radius *= voxel_physical_size;
                spawn_radius *= voxel_physical_size;

                let mut recurse = true;
                let mut spawn = false;
                let mut despawn = false;

                match (dist < spawn_radius,) {
                    (true,) => {
                        spawn = true;
                    }
                    (false,) => {
                        recurse = false;
                        despawn = true;
                    }
                }

                chunks
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(i, c)| c.as_mut().map(|c| (i, c)))
                    .for_each(|(i, c)| {
                        let mut ipos = chunk_pos;
                        let mut offset = IVec3::ONE * size;
                        if i & 0b001 == 0 {
                            offset.x *= -1;
                        }
                        if i & 0b010 == 0 {
                            offset.y *= -1;
                        }
                        if i & 0b100 == 0 {
                            offset.z *= -1;
                        }
                        ipos += offset;

                        let pos = ipos.as_vec3() * voxel_physical_size;
                        let dir = (pos - player).normalize() * voxel_physical_size;
                        let close = pos - size as f32 * dir;
                        let far = pos + size as f32 * dir;
                        let close_dist = (close - player).length().abs();
                        let far_dist = (far - player).length().abs();
                        let center_dist = (pos - player).length().abs();
                        let dist = center_dist.min(close_dist);

                        let r = 128.0;
                        let mult = 2u32.pow(height - chunk_height) as f32;
                        let mut despawn_radius = r * 6.0 * mult;
                        let mut spawn_radius = r * 2.0 * mult;
                        despawn_radius *= voxel_physical_size;
                        spawn_radius *= voxel_physical_size;

                        let mut recurse = recurse;
                        let mut spawn = spawn;
                        let mut despawn = despawn;

                        match (dist < spawn_radius, height == chunk_height + 1) {
                            (_, true) => {}
                            (true, false) => {
                                spawn = false;
                                despawn = true;
                                recurse = true;
                            }
                            (false, false) => {
                                // recurse = false;
                            }
                        }

                        if spawn {
                            c.set_mip_visibility(commands, Visibility::Inherited);
                        }
                        if despawn {
                            c.set_mip_visibility(commands, Visibility::Hidden);
                        }
                        if recurse {
                            c.update_chunks(
                                player,
                                ipos,
                                height - 1,
                                chunk_height,
                                svo_entity,
                                commands,
                                images,
                                chunk_materials,
                                voxel_physical_size,
                                materials,
                                cube_mesh,
                            );
                        }
                    });
            }
            Self::Chunk { voxels } => {}
        }
    }

    // TODO: very inefficient
    fn update_mip(&mut self, chunk_height: u32) {
        match self {
            Self::Chunk { voxels } => {
                voxels.update_mips();
            }
            Self::Node { chunks, mip } => {
                chunks
                    .iter_mut()
                    .flatten()
                    .for_each(|c| c.update_mip(chunk_height));
                let size = 2usize.pow(chunk_height);
                let mut byte = ByteChunk::new_with_size(2 * size);

                // let size = size / 2;
                let size = size as _;
                chunks
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| c.as_ref().map(|c| (i, c.mip())))
                    .for_each(|(i, c)| {
                        let mut offset = UVec3::ZERO;
                        if i & 0b001 > 0 {
                            offset.x = size;
                        }
                        if i & 0b010 > 0 {
                            offset.y = size;
                        }
                        if i & 0b100 > 0 {
                            offset.z = size;
                        }
                        for z in 0..size {
                            for y in 0..size {
                                for x in 0..size {
                                    let pos = UVec3::new(x, y, z);
                                    byte.set(pos + offset, *c.byte.get(pos).unwrap());
                                }
                            }
                        }
                    });

                let byte = byte.mip1_bytechunk();
                let mip1 = byte.mip();
                let mip2 = mip1.mip();
                let chunk = Chunk {
                    byte,
                    mip1,
                    mip2,
                    entity: None,
                };
                *mip = chunk;
            }
        }
    }

    // does not update the mips
    fn set_voxel(&mut self, pos: IVec3, voxel: u8, height: u32, chunk_height: u32) -> u8 {
        match self {
            Self::Node { chunks, .. } => {
                assert!(height > chunk_height);
                let mut index = 0;
                let mut offset = IVec3::ONE * 2i32.pow(height - 2);
                if pos.z >= 0 {
                    index |= 0b100;
                } else {
                    offset.z *= -1;
                }
                if pos.y >= 0 {
                    index |= 0b010;
                } else {
                    offset.y *= -1;
                }
                if pos.x >= 0 {
                    index |= 0b001;
                } else {
                    offset.x *= -1;
                }
                // dbg!(pos, offset, pos-offset);
                chunks[index]
                    .get_or_insert_with(|| {
                        if height - 1 == chunk_height {
                            Self::new_leaf(chunk_height)
                        } else {
                            Self::new_node(chunk_height)
                        }
                    })
                    .set_voxel(pos - offset, voxel, height - 1, chunk_height)
            }
            Self::Chunk { voxels } => {
                assert_eq!(height, chunk_height);
                let offset = IVec3::ONE * 2i32.pow(chunk_height - 1);
                // dbg!(pos, 2i32.pow(chunk_height - 1), pos + offset);
                let pos = pos + offset;
                let pos = pos.try_into().unwrap();
                voxels.byte.set(pos, voxel).unwrap()
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_chunks(
        &mut self,
        svo_entity: Entity,
        commands: &mut Commands,
        images: &mut Assets<Image>,
        chunk_materials: &mut Assets<ChunkMaterial>,
        voxel_physical_size: f32,
        materials: &Handle<Image>,
        cube_mesh: &Handle<Mesh>,
        chunk_pos: IVec3,
        height: u32,
    ) {
        match self {
            Self::Node { chunks, .. } => {
                chunks
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(i, c)| c.as_mut().map(|c| (i, c)))
                    .for_each(|(i, c)| {
                        let mut offset = IVec3::ONE * 2i32.pow(height - 2);
                        if i & 0b100 == 0 {
                            offset.z *= -1;
                        }
                        if i & 0b010 == 0 {
                            offset.y *= -1;
                        }
                        if i & 0b001 == 0 {
                            offset.x *= -1;
                        }
                        let pos = chunk_pos + offset;
                        c.spawn_chunks(
                            svo_entity,
                            commands,
                            images,
                            chunk_materials,
                            voxel_physical_size,
                            materials,
                            cube_mesh,
                            pos,
                            height - 1,
                        );
                    });
            }
            Self::Chunk { voxels } => {
                voxels.spawn(
                    svo_entity,
                    commands,
                    images,
                    chunk_materials,
                    voxel_physical_size,
                    materials,
                    cube_mesh,
                    chunk_pos.as_vec3() * voxel_physical_size,
                );
            }
        }
    }
}

pub struct TiledChunker {
    pub chunk_side: usize,
    pub chunks: HashMap<IVec3, ByteChunk>,
}

impl TiledChunker {
    pub fn with_chunk_size(size: usize) -> Self {
        Self {
            chunk_side: size,
            chunks: Default::default(),
        }
    }

    // https://www.desmos.com/calculator/dcrznftoqg
    pub fn get_voxel_index(&self, mut index: IVec3) -> UVec3 {
        let side = self.chunk_side as i32;
        // if index.x%side == 0 || index.z%side == 0 {
        //     let index = ((index % side) + side) % side;
        //     return  UVec3::new(side as u32 - 1, index.y as _, side as u32 - 1);
        // }
        let index = index.signum().min(IVec3::splat(0)) + index;
        let index = ((index % side) + side) % side;
        // let index = Vec3::new(index.x as _, index.y as _, index.z as _) % side as f32;
        // let index = index + side as f32;
        // let index = index % side as f32;
        UVec3::new(index.x as _, index.y as _, index.z as _)
    }
    pub fn get_chunk_index(&self, index: IVec3) -> IVec3 {
        let side = self.chunk_side as i32;
        let negative_offset = index.signum().min(IVec3::new(0, 0, 0));
        // floor((-1, -1)/2) gives (-1, -1)
        // but only in floating point math.
        // we need to subtract 1 in the directions with -ve values when working with integers
        negative_offset + index / side
        // let index = index.signum().min(IVec3::splat(0)) + index;
        // index / side
    }

    pub fn set_voxel(&mut self, index: IVec3, voxel: u8) -> Option<u8> {
        // let index = index.signum().min(IVec3::splat(0)) + index;
        let i1 = self.get_chunk_index(index);
        let index = self.get_voxel_index(index);
        match self.chunks.get_mut(&i1) {
            Some(c) => c.set(index, voxel),
            None => {
                let mut c = ByteChunk::new_with_size(self.chunk_side);
                let v = c.set(index, voxel);
                self.chunks.insert(i1, c);
                v
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_chunks(
        self,
        scene_entity: Entity,
        commands: &mut Commands,
        images: &mut Assets<Image>,
        chunk_materials: &mut Assets<ChunkMaterial>,
        voxel_physical_size: f32,
        materials: &Handle<Image>,
        cube_mesh: &Handle<Mesh>,
        chunk_pos: IVec3,
    ) {
        for (pos, bytechunk) in self.chunks {
            let mip1 = bytechunk.mip();
            let mip2 = mip1.mip();

            let mut chunk = Chunk {
                byte: bytechunk,
                mip1,
                mip2,
                entity: None,
            };

            chunk.spawn(
                scene_entity,
                commands,
                images,
                chunk_materials,
                voxel_physical_size,
                materials,
                cube_mesh,
                (pos + chunk_pos).as_vec3() * self.chunk_side as f32 * voxel_physical_size,
            );
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StaticChunkOctreeNode<T> {
    pub side: usize,
    pub subchunks: [Option<T>; 8],
}

pub type L1ChunkOctree = StaticChunkOctreeNode<ByteChunk>;
pub type L2ChunkOctree = StaticChunkOctreeNode<StaticChunkOctreeNode<ByteChunk>>;

impl L1ChunkOctree {
    pub fn set_voxel(&mut self, index: UVec3, voxel: u8) -> Option<u8> {
        let hside = self.side as u32 / 2;
        let i1 = UVec3::new(index.x % hside, index.y % hside, index.z % hside);
        match self.get_mut(index) {
            Some(c) => {
                let c = c.get_or_insert_with(|| ByteChunk::new_with_size(hside as _));
                c.set(i1, voxel)
            }
            None => None,
        }
    }
}
impl L2ChunkOctree {
    pub fn set_voxel(&mut self, index: UVec3, voxel: u8) -> Option<u8> {
        let hside = self.side as u32 / 2;
        let i1 = UVec3::new(index.x % hside, index.y % hside, index.z % hside);
        match self.get_mut(index) {
            Some(c) => {
                let c = c.get_or_insert_with(|| L1ChunkOctree::new_with_size(hside as _));
                c.set_voxel(i1, voxel)
            }
            None => None,
        }
    }
}

pub trait ChunkTrait {
    type Voxel;
    fn new_with_size(size: usize) -> Self;
    fn get_voxels(&self) -> &[Self::Voxel];
    fn get_voxels_mut(&mut self) -> &mut [Self::Voxel];
    fn get_index(&self, index: UVec3) -> Option<usize>;

    fn get_mut(&mut self, index: UVec3) -> Option<&mut Self::Voxel> {
        let index = self.get_index(index);
        index.map(|i| &mut self.get_voxels_mut()[i])
    }
    fn get(&self, index: UVec3) -> Option<&Self::Voxel> {
        let index = self.get_index(index);
        index.map(|i| &self.get_voxels()[i])
    }
    // returns old voxel at that position
    // returns None if index is outside the chunk
    fn set(&mut self, index: UVec3, voxel: Self::Voxel) -> Option<Self::Voxel>
    where
        Self::Voxel: Copy,
    {
        // if index.max_element() > self.side as _ {
        //     return None;
        // }
        // let side = self.side as u32;
        // let index = side.pow(2) * index.z + side * index.y + index.x;
        // let index = index as usize;
        // let old_voxel = self.voxels[index];
        // self.voxels[index] = voxel;
        // Some(old_voxel)
        match self.get_mut(index) {
            Some(old) => {
                let old_voxel = *old;
                *old = voxel;
                Some(old_voxel)
            }
            None => None,
        }
    }
}

impl<T> ChunkTrait for StaticChunkOctreeNode<T> {
    type Voxel = Option<T>;

    fn new_with_size(side: usize) -> Self {
        Self {
            side,
            subchunks: Default::default(),
        }
    }

    fn get_voxels(&self) -> &[Self::Voxel] {
        &self.subchunks
    }

    fn get_voxels_mut(&mut self) -> &mut [Self::Voxel] {
        &mut self.subchunks
    }

    #[allow(clippy::identity_op)]
    fn get_index(&self, index: UVec3) -> Option<usize> {
        let side = self.side as _;
        if index.max_element() >= side {
            return None;
        }
        let hside = side / 2;
        let index = UVec3::new(
            ((index.z > hside) as u32) << 0b100,
            ((index.y > hside) as u32) << 0b010,
            ((index.x > hside) as u32) << 0b000,
        );
        let index = side.pow(2) * index.z + side * index.y + index.x;
        Some(index as _)
    }
}

impl ChunkTrait for ByteChunk {
    type Voxel = u8;

    fn new_with_size(side: usize) -> Self {
        Self {
            side,
            voxels: vec![0; side.pow(3)],
        }
    }

    fn get_voxels(&self) -> &[Self::Voxel] {
        &self.voxels
    }

    fn get_voxels_mut(&mut self) -> &mut [Self::Voxel] {
        &mut self.voxels
    }

    fn get_index(&self, index: UVec3) -> Option<usize> {
        let side = self.side as _;
        if index.max_element() >= side {
            return None;
        }
        let index = side.pow(2) * index.z + side * index.y + index.x;
        Some(index as _)
    }
}

#[derive(Clone, Debug)]
pub struct U8VoxelChunk {
    pub voxels: Vec<U8Voxel>,
    pub side: usize,
}

impl U8VoxelChunk {
    pub fn byte_chunk(&self) -> ByteChunk {
        let side = self.side;
        let pside = side + 2;
        let mut new_voxels = vec![0; side.pow(3)];
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    new_voxels[side.pow(2) * z + side * y + x] =
                        self.voxels[pside.pow(2) * (z + 1) + pside * (y + 1) + (x + 1)].0;
                }
            }
        }

        ByteChunk {
            side,
            voxels: new_voxels,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Chunk {
    byte: ByteChunk,
    mip1: MipChunk,
    mip2: MipChunk,
    entity: Option<Entity>,
}
impl Chunk {
    fn update_mips(&mut self) {
        self.mip1 = self.byte.mip();
        self.mip2 = self.mip1.mip();
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &mut self,
        svo_entity: Entity,
        commands: &mut Commands,
        images: &mut Assets<Image>,
        chunk_materials: &mut Assets<ChunkMaterial>,
        voxel_physical_size: f32,
        materials: &Handle<Image>,
        cube_mesh: &Handle<Mesh>,
        chunk_pos: Vec3,
    ) {
        let side = self.byte.side;

        let voxels_handle = images.add(self.byte.clone().to_image());
        let mip1_handle = images.add(self.mip1.clone().into_image());
        let mip2_handle = images.add(self.mip2.clone().into_image());

        let chunk_material = chunk_materials.add(ChunkMaterial {
            side: side as _,
            voxels: voxels_handle.clone(),
            materials: materials.clone(),
            chunk_position: chunk_pos,
            voxel_size: voxel_physical_size,
            voxels_mip1: mip1_handle,
            voxels_mip2: mip2_handle,
        });

        commands.entity(svo_entity).with_children(|builder| {
            let id = builder
                .spawn((
                    Name::new("VoxChunk"),
                    vox::VoxChunk,
                    MaterialMeshBundle {
                        mesh: cube_mesh.clone(),
                        material: chunk_material,
                        transform: Transform::from_translation(chunk_pos)
                            .with_scale(Vec3::NEG_ONE * voxel_physical_size * side as f32 / 2.0),
                        ..Default::default()
                    },
                ))
                .id();
            self.entity = Some(id);
        });
    }
}

#[derive(Clone, Debug)]
pub struct ByteChunk {
    pub voxels: Vec<u8>,
    pub side: usize,
}

impl ByteChunk {
    pub fn mip1_bytechunk(&self) -> Self {
        let tc = TChunk::mip1_tchunk_from_slice(&self.voxels, self.side);
        Self {
            voxels: tc.voxels,
            side: tc.side,
        }
    }

    pub fn mip(&self) -> MipChunk {
        TChunk::mip_from_slice(&self.voxels, self.side)
    }

    pub fn to_image(self) -> Image {
        Image::new(
            Extent3d {
                width: self.side as _,
                height: self.side as _,
                depth_or_array_layers: self.side as _,
            },
            TextureDimension::D3,
            self.voxels,
            TextureFormat::R8Uint,
        )
    }
}

#[derive(Clone, Debug)]
pub struct TChunk<T> {
    pub voxels: Vec<T>,
    pub side: usize,
}

impl<T> TChunk<T>
where
    T: From<u8> + Copy + PartialEq + PartialOrd + AddAssign,
{
    pub fn mip1_tchunk(&self) -> Self {
        Self::mip1_tchunk_from_slice(&self.voxels, self.side)
    }

    #[allow(clippy::identity_op)]
    pub fn mip1_tchunk_from_slice(voxels: &[T], side: usize) -> Self {
        assert_eq!(side % 2, 0, "side should be divisible by 2");
        let new_side = side / 2;
        let mut bc = Self {
            voxels: vec![0u8.into(); new_side.pow(3)],
            side: new_side,
        };
        let voxel = |z, y, x| voxels[z * side.pow(2) + y * side + x];

        for z in 0..new_side {
            for y in 0..new_side {
                for x in 0..new_side {
                    let mut voxels: [(T, T); 8] = [(0u8.into(), 0u8.into()); 8];
                    let samples = [
                        voxel(z * 2 + 0, y * 2 + 0, x * 2 + 0),
                        voxel(z * 2 + 0, y * 2 + 0, x * 2 + 1),
                        voxel(z * 2 + 0, y * 2 + 1, x * 2 + 0),
                        voxel(z * 2 + 0, y * 2 + 1, x * 2 + 1),
                        voxel(z * 2 + 1, y * 2 + 0, x * 2 + 0),
                        voxel(z * 2 + 1, y * 2 + 0, x * 2 + 1),
                        voxel(z * 2 + 1, y * 2 + 1, x * 2 + 0),
                        voxel(z * 2 + 1, y * 2 + 1, x * 2 + 1),
                    ];
                    for sample in samples {
                        for i in 0..8 {
                            if voxels[i].0 == 0u8.into() {
                                voxels[i] = (sample, 1u8.into());
                                break;
                            } else if voxels[i].0 == sample {
                                voxels[i].1 += 1u8.into();
                                if voxels[i].1 > voxels[0].1 {
                                    voxels.swap(0, i);
                                }
                                break;
                            }
                        }
                    }
                    bc.voxels[bc.side.pow(2) * z + bc.side * y + x] = voxels[0].0;
                }
            }
        }

        bc
    }

    pub fn mip(&self) -> MipChunk {
        Self::mip_from_slice(&self.voxels, self.side)
    }

    // chunk side should be a multiple of 4
    // 1 byte stores 2x2x2 voxels each a single bit
    // 1 vec2<u32> stores 2x2x2 bytes, so 4x4x4 voxels
    // can use TextureFormat::Rg32Uint
    #[allow(clippy::erasing_op, clippy::identity_op)]
    pub fn mip_from_slice(voxels: &[T], side: usize) -> MipChunk {
        let new_side = side / 4;
        assert_eq!(side % 4, 0, "side should be a multiple of 4");

        let voxel = |z, y, x| (voxels[z * side.pow(2) + y * side + x] != 0u8.into()) as u32;
        let byte = |z, y, x| {
            000 | voxel(z + 0, y + 0, x + 0) << 0b000
                | voxel(z + 0, y + 0, x + 1) << 0b001
                | voxel(z + 0, y + 1, x + 0) << 0b010
                | voxel(z + 0, y + 1, x + 1) << 0b011
                | voxel(z + 1, y + 0, x + 0) << 0b100
                | voxel(z + 1, y + 0, x + 1) << 0b101
                | voxel(z + 1, y + 1, x + 0) << 0b110
                | voxel(z + 1, y + 1, x + 1) << 0b111
        };
        let chunk1 = |z, y, x| {
            000 | byte(z * 4 + 0, y * 4 + 0, x * 4 + 0) << (0b000 * 8)
                | byte(z * 4 + 0, y * 4 + 0, x * 4 + 2) << (0b001 * 8)
                | byte(z * 4 + 0, y * 4 + 2, x * 4 + 0) << (0b010 * 8)
                | byte(z * 4 + 0, y * 4 + 2, x * 4 + 2) << (0b011 * 8)
        };
        let chunk2 = |z, y, x| {
            000 | byte(z * 4 + 2, y * 4 + 0, x * 4 + 0) << (0b000 * 8)
                | byte(z * 4 + 2, y * 4 + 0, x * 4 + 2) << (0b001 * 8)
                | byte(z * 4 + 2, y * 4 + 2, x * 4 + 0) << (0b010 * 8)
                | byte(z * 4 + 2, y * 4 + 2, x * 4 + 2) << (0b011 * 8)
        };

        let mut mip = vec![UVec2::ZERO; new_side.pow(3)];
        for z in 0..new_side {
            for y in 0..new_side {
                for x in 0..new_side {
                    mip[new_side.pow(2) * z + new_side * y + x] =
                        UVec2::new(chunk1(z, y, x), chunk2(z, y, x));
                }
            }
        }
        MipChunk { voxels: mip, side }
    }
}

pub struct BitWorld {
    // TODO: an offset for sliding window chunk view
    // chunk_offset: UVec2,
    // 2^5 cubed for now
    // index into 'data' where a chunk starts
    pub chunk_indices: Vec<u32>,

    // voxels at the highest mip level
    pub chunk_side: usize,
    pub chunk_buffer: Vec<UVec2>,
}
impl BitWorld {
    pub fn new() -> Self {
        let chunk_indices = vec![0; 2usize.pow(5).pow(3)];
        Self {
            chunk_buffer: TChunk::mip_from_slice(&chunk_indices, 2usize.pow(5)).voxels,
            chunk_indices,
            chunk_side: 128,
            // chunk_buffer: vec![UVec2::ZERO], // '0' as index is not allowed in 'chunk_indices'
        }
    }

    pub fn insert(&mut self, mip: &MipChunk, pos: UVec3) {
        let side = 2u32.pow(5);
        self.insert_at_index(mip, pos.z * side.pow(2) + pos.y * side + pos.x);
    }

    pub fn insert_at_index(&mut self, mip: &MipChunk, index: u32) {
        self.chunk_indices[index as usize] = self.chunk_buffer.len() as _;
        self.chunk_buffer.extend_from_slice(&mip.voxels);
    }

    pub fn overwrite_chunk_at_index(&mut self, mip: &MipChunk, index: u32) {
        let i = self.chunk_indices[index as usize];
        self.chunk_buffer[i as usize..i as usize + mip.voxels.len()].copy_from_slice(&mip.voxels);
    }

    pub fn mip(&self) -> MipChunk {
        TChunk::mip_from_slice(&self.chunk_indices, 2usize.pow(5))
    }

    pub fn update_mip(&mut self) {
        let tc = self.mip();
        self.overwrite_chunk_at_index(&tc, 0);
    }
}

#[derive(Clone, Debug)]
pub struct MipChunk {
    // voxels at the highest mip level
    pub side: usize,

    // length of this should always be (side/4)^3
    // each UVec2 stores 4x4x4 voxels
    pub voxels: Vec<UVec2>,
}

impl MipChunk {
    pub fn mip(&self) -> Self {
        Self::mip_from_slice(&self.voxels, self.side)
    }

    // from mip (0, 1, 2) to (3, 4, 5)
    // 0 -> 1 bit
    // 1 -> 1 byte, 2x2x2 voxels
    // 2 -> 2 x u32, 4x4x4 voxels
    #[allow(clippy::erasing_op, clippy::identity_op)]
    pub fn mip_from_slice(voxels: &[UVec2], side: usize) -> Self {
        let mip_side = side / 4;
        let new_side = mip_side / 8;
        assert_eq!(side % (4 * 8), 0, "side should be a multiple of 32");

        let bit = |z, y, x| {
            let chunk: &UVec2 = &voxels[z * mip_side.pow(2) + y * mip_side + x];
            chunk.x != 0 || chunk.y != 0
        };
        let voxel = |z, y, x| {
            000 | (bit(z + 0, y + 0, x + 0)
                || bit(z + 0, y + 0, x + 1)
                || bit(z + 0, y + 1, x + 0)
                || bit(z + 0, y + 1, x + 1)
                || bit(z + 1, y + 0, x + 0)
                || bit(z + 1, y + 0, x + 1)
                || bit(z + 1, y + 1, x + 0)
                || bit(z + 1, y + 1, x + 1)) as u32
        };
        let byte = |z, y, x| {
            000 | voxel(z + 0, y + 0, x + 0) << 0b000
                | voxel(z + 0, y + 0, x + 2) << 0b001
                | voxel(z + 0, y + 2, x + 0) << 0b010
                | voxel(z + 0, y + 2, x + 2) << 0b011
                | voxel(z + 2, y + 0, x + 0) << 0b100
                | voxel(z + 2, y + 0, x + 2) << 0b101
                | voxel(z + 2, y + 2, x + 0) << 0b110
                | voxel(z + 2, y + 2, x + 2) << 0b111
        };
        let chunk1 = |z, y, x| {
            000 | byte(z * 8 + 0, y * 8 + 0, x * 8 + 0) << (0b000 * 8)
                | byte(z * 8 + 0, y * 8 + 0, x * 8 + 4) << (0b001 * 8)
                | byte(z * 8 + 0, y * 8 + 4, x * 8 + 0) << (0b010 * 8)
                | byte(z * 8 + 0, y * 8 + 4, x * 8 + 4) << (0b011 * 8)
        };
        let chunk2 = |z, y, x| {
            000 | byte(z * 8 + 4, y * 8 + 0, x * 8 + 0) << (0b000 * 8)
                | byte(z * 8 + 4, y * 8 + 0, x * 8 + 4) << (0b001 * 8)
                | byte(z * 8 + 4, y * 8 + 4, x * 8 + 0) << (0b010 * 8)
                | byte(z * 8 + 4, y * 8 + 4, x * 8 + 4) << (0b011 * 8)
        };

        let mut mip = vec![UVec2::ZERO; new_side.pow(3)];
        for z in 0..new_side {
            for y in 0..new_side {
                for x in 0..new_side {
                    mip[new_side.pow(2) * z + new_side * y + x] =
                        UVec2::new(chunk1(z, y, x), chunk2(z, y, x));
                }
            }
        }
        MipChunk {
            voxels: mip,
            side: mip_side / 2,
        }
    }

    pub fn into_image(self) -> Image {
        let mip_side = self.side / 4;
        Image::new(
            Extent3d {
                width: mip_side as _,
                height: mip_side as _,
                depth_or_array_layers: mip_side as _,
            },
            TextureDimension::D3,
            self.voxels
                .into_iter()
                .flat_map(|v| [v.x.to_le_bytes(), v.y.to_le_bytes()])
                .flatten()
                .collect(),
            TextureFormat::Rg32Uint,
        )
    }
}

pub const DEFAULT_CHUNK_SIDE: u32 = 8 * 16;
pub const PADDED_DEFAULT_CHUNK_SIDE: u32 = 8 * 16 + 2;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct U8Voxel(u8);
impl Voxel for U8Voxel {
    fn get_visibility(&self) -> VoxelVisibility {
        if self.0 == 0 {
            VoxelVisibility::Empty
        } else {
            VoxelVisibility::Opaque
        }
    }
}
impl MergeVoxel for U8Voxel {
    type MergeValue = Self;

    fn merge_value(&self) -> Self::MergeValue {
        *self
    }
}

#[derive(Component, Clone, Debug)]
pub struct ChunkHandle {
    // TODO: turn these to have a runtime parameter of size: UVec3
    pub side: usize,
    pub voxels: Handle<Image>,
    pub materials: Handle<Image>,
}

/// N should be a multiple of 4
impl ChunkHandle {
    pub fn empty(assets: &mut Assets<Image>) -> Self {
        Self {
            voxels: assets.add(
                ByteChunk {
                    voxels: vec![0; DEFAULT_CHUNK_SIDE.pow(3) as _],
                    side: DEFAULT_CHUNK_SIDE as _,
                }
                .to_image(),
            ),
            materials: assets.add(Self::material_image(vec![Default::default(); 256])),
            side: DEFAULT_CHUNK_SIDE as _,
        }
    }

    pub fn material_image(materials: Vec<Vec4>) -> Image {
        Image::new(
            Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D1,
            materials
                .into_iter()
                .flat_map(|v| [v.x, v.y, v.z, v.w])
                .map(|v| (v * 255.0) as u8)
                .collect(),
            TextureFormat::Rgba8Unorm, // Rgba8UnormSrgb is gamma-corrected
        )
    }
}

// - [AsBindGroup in bevy::render::render_resource - Rust](https://docs.rs/bevy/latest/bevy/render/render_resource/trait.AsBindGroup.html)
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Component, ExtractComponent)]
pub struct ChunkMaterial {
    #[uniform(0)]
    pub side: u32,
    #[texture(2, dimension = "1d", sample_type = "float", filterable = false)]
    pub materials: Handle<Image>,
    #[uniform(5)]
    pub chunk_position: Vec3,
    #[uniform(6)]
    pub voxel_size: f32,
    #[texture(1, sample_type = "u_int", dimension = "3d")]
    pub voxels: Handle<Image>,
    #[texture(7, sample_type = "u_int", dimension = "3d")]
    pub voxels_mip1: Handle<Image>,
    #[texture(8, sample_type = "u_int", dimension = "3d")]
    pub voxels_mip2: Handle<Image>,
}

impl Material for ChunkMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/chunk.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

#[derive(Component)]
pub struct ChunkSpawnTask {
    task: Task<((U8VoxelChunk, Vec3, f32), Mesh)>,
}

fn setup_voxel_plugin(mut commands: Commands, mut game_state: ResMut<NextState<AppState>>) {
    let perlin = noise::Perlin::new(3243);
    let world = VoxelWorld { noise: perlin };
    commands.insert_resource(world);
    game_state.set(AppState::Playing);
}

/*
fn spawn_chunk_tasks(
    mut commands: Commands,
    world: Res<VoxelWorld>,
    mut svo: Query<(Entity, &mut SparseVoxelOctree)>,
    players: Query<(&PlayerEntity, &Transform)>,
) {
    let threadpool = AsyncComputeTaskPool::get();

    let (_, player) = players.single();

    for (svo_entity, mut svo) in svo.iter_mut() {
        let Some(chunk) = svo.get_mut_chunk(player.translation) else {
            return;
        };
        dbg!("spawining shunk", chunk.pos, chunk.size);

        let size = chunk.size;
        let pos = chunk.pos;
        let perlin = world.noise;
        let task = threadpool.spawn(async move {
            let side = DEFAULT_CHUNK_SIDE as usize;
            let pside = PADDED_DEFAULT_CHUNK_SIDE as usize;
            let mut voxels = U8VoxelChunk(vec![U8Voxel(0); pside * pside * pside]);
            let scale = 0.09;
            for z in 0..side {
                for x in 0..side {
                    let xf =
                        ((x as f32 - side as f32 / 2.0) / side as f32) * size * 2.0 + pos.x;
                    let zf =
                        ((z as f32 - side as f32 / 2.0) / side as f32) * size * 2.0 + pos.z;
                    let v = perlin.get([xf as f64 * scale, zf as f64 * scale]);

                    for y in 0..side {
                        let yf =
                            ((y as f32 - side as f32 / 2.0) / side as f32) * size * 2.0 + pos.y;
                        if (yf as f64 + 20.0) < v * 10.0 {
                            voxels.0[(z + 1) * pside * pside + (y + 1) * pside + (x + 1)] =
                                U8Voxel(1);
                        }
                    }
                }
            }
            let mut buffer = GreedyQuadsBuffer::new(voxels.0.len());
            type ChunkShape = ConstShape3u32<
                PADDED_DEFAULT_CHUNK_SIDE,
                PADDED_DEFAULT_CHUNK_SIDE,
                PADDED_DEFAULT_CHUNK_SIDE,
            >;
            let faces = &block_mesh::RIGHT_HANDED_Y_UP_CONFIG.faces;
            greedy_quads(
                &voxels.0,
                &ChunkShape {},
                [0; 3],
                [DEFAULT_CHUNK_SIDE + 1; 3],
                faces,
                &mut buffer,
            );

            let num_indices = buffer.quads.num_quads() * 6;
            let num_vertices = buffer.quads.num_quads() * 4;
            let mut indices = Vec::with_capacity(num_indices);
            let mut positions = Vec::with_capacity(num_vertices);
            let mut normals = Vec::with_capacity(num_vertices);
            for (group, face) in buffer.quads.groups.into_iter().zip(faces.iter()) {
                for quad in group.into_iter() {
                    indices.extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));
                    positions.extend_from_slice(
                        &face
                            .quad_mesh_positions(&quad, 2.0 * size / DEFAULT_CHUNK_SIDE as f32),
                    );
                    normals.extend_from_slice(&face.quad_mesh_normals());
                }
            }

            let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
            render_mesh.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(positions),
            );
            render_mesh.insert_attribute(
                Mesh::ATTRIBUTE_NORMAL,
                VertexAttributeValues::Float32x3(normals),
            );
            render_mesh.insert_attribute(
                Mesh::ATTRIBUTE_UV_0,
                VertexAttributeValues::Float32x2(vec![[0.0; 2]; num_vertices]),
            );
            render_mesh.set_indices(Some(Indices::U32(indices.clone())));

            ((voxels, pos, size), render_mesh)
        });

        *chunk.chunk_entity = commands.spawn_empty().id();
        commands
            .entity(*chunk.chunk_entity)
            .insert(ChunkSpawnTask { task });

        commands.entity(svo_entity).add_child(*chunk.chunk_entity);
    }
}

fn resolve_chunk_tasks<const N: usize>(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut chunk_materials: ResMut<Assets<ChunkMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut tasks: Query<(Entity, &mut ChunkSpawnTask<N>)>,
) {
    for (task_entity, mut task) in tasks.iter_mut() {
        // tasks can be cancelled. so it returns an Option
        if let Some(((voxels, pos, size), mesh)) =
            block_on(futures_lite::future::poll_once(&mut task.task))
        {
            let empty_mesh = mesh.count_vertices() == 0;

            let mesh_handle = meshes.add(mesh);
            let mut material_buffer = vec![Default::default(); 256];
            material_buffer[1] = Vec4::new(0.8, 0.8, 0.8, 1.0);
            // let chunk = DefaultChunk::from_u8voxels(&mut images, voxels, material_buffer);
            let chunk = DefaultChunk {
                voxels: images.add(voxels.byte_chunk().to_image()),
                materials: images.add(DefaultChunk::material_image(material_buffer)),
            };

            let r = (DEFAULT_CHUNK_SIDE as f32 + 2.0) / (DEFAULT_CHUNK_SIDE as f32);
            let bb_mesh_relative_pos = Vec3::ZERO;
            let voxel_mesh_relative_pos = Vec3::splat(-size * r);
            commands
                .entity(task_entity)
                .insert((
                    TransformBundle::from_transform(Transform::from_translation(pos)),
                    VisibilityBundle::default(),
                ))
                .remove::<ChunkSpawnTask<N>>();

            if empty_mesh {
                continue;
            }

            commands
                .entity(task_entity)
                .with_children(|parent| {
                    parent.spawn((
                        MaterialMeshBundle {
                            mesh: meshes.add(Mesh::from(shape::Cube { size: size * 2.0 })),
                            material: chunk_materials.add(ChunkMaterial {
                                side: DEFAULT_CHUNK_SIDE,
                                voxels: chunk.voxels.clone(),
                                materials: chunk.materials.clone(),
                                player_position: Vec3::ZERO,
                                resolution: Vec2::ZERO,
                                chunk_position: Vec3::ZERO,
                                chunk_size: size,
                            }),
                            visibility: Visibility::Visible,
                            transform: Transform::from_translation(bb_mesh_relative_pos),
                            ..Default::default()
                        },
                        Name::new("Cube Mesh"),
                    ));
                    parent.spawn((
                        MaterialMeshBundle {
                            mesh: mesh_handle.clone(),
                            // material: materials.add(StandardMaterial {
                            //     base_color: Color::rgb(0.8, 0.8, 0.8),
                            //     alpha_mode: AlphaMode::Opaque,
                            //     ..Default::default()
                            // }),
                            material: chunk_materials.add(ChunkMaterial {
                                side: DEFAULT_CHUNK_SIDE,
                                voxels: chunk.voxels.clone(),
                                materials: chunk.materials.clone(),
                                player_position: Vec3::ZERO,
                                resolution: Vec2::ZERO,
                                chunk_position: Vec3::ZERO,
                                chunk_size: size,
                            }),
                            transform: Transform::from_translation(voxel_mesh_relative_pos),
                            visibility: Visibility::Hidden,
                            ..Default::default()
                        },
                        Name::new("Voxel Mesh"),
                    ));
                })
                .insert((Name::new("Chunk"), chunk));
        }
    }
}
*/
