#import bevy_pbr::forward_io::VertexOutput

@group(1) @binding(0) var<uniform> side: u32;
@group(1) @binding(5) var<uniform> chunk_pos: vec3<f32>;
@group(1) @binding(6) var<uniform> voxel_size: f32;

@group(1) @binding(2) var materials: texture_1d<f32>;

struct WorldDataUniforms {
    player_pos: vec3<f32>,
    screen_resolution: vec2<u32>,
    fov: f32,
    unit_t_max: f32,
    depth_prepass_scale_factor: u32,
}

@group(3) @binding(0) var<uniform> world_data: WorldDataUniforms;
@group(3) @binding(1) var transient_world_texture: texture_storage_3d<r32uint, read_write>;

// returns voxel material in the rightmost byte
fn get_voxel(pos: vec3<f32>) -> u32 {
    if (any(pos >= f32(side) || pos < 0.0)) {
        return 0u;
    }
    
    // loading textures automatically break the u32 into a vec4<u32> 1 byte each channel (~~ vec4<u8>)
    var voxel = textureLoad(transient_world_texture, vec3<i32>(pos)).x;

    return voxel;
}

fn get_color(index: u32) -> vec4<f32> {
    return textureLoad(materials, index, 0);
}


fn unpackBytes(t: u32) -> vec4<u32> {
    return (t >> vec4<u32>(0u, 8u, 16u, 24u)) & 15u;
}

fn getMipByte(mip: vec2<u32>, index: u32) -> u32 {
    var component = 0u;
    switch (index) {
        case 0u {
            component = mip.x & 255u;
        }
        case 1u {
            component = mip.x & (255u << 8u);
        }
        case 2u {
            component = mip.x & (255u << 16u);
        }
        case 3u {
            component = mip.x & (255u << 24u);
        }
        case 4u {
            component = mip.y & 255u;
        }
        case 5u {
            component = mip.y & (255u << 8u);
        }
        case 6u {
            component = mip.y & (255u << 16u);
        }
        case 7u {
            component = mip.y & (255u << 24u);
        }
        default: {
            // discard;
        }
    }
    return component;
}
fn getMipBit(mip: u32, index: u32) -> u32 {
    var mask = 1u << index;
    mask = mask | mask << 8u | mask << 16u | mask << 24u;
    return mip & mask;
}

struct Output {
    @location(0) color: vec4<f32>,
}

@fragment
fn fragment(mesh: VertexOutput) -> Output {
    var out: Output;

    if true {
        // out.color =  vec4<f32>((mesh.world_position.xyz - chunk_pos)/1000.0, 1.0);
        // out.color = vec4<f32>(mesh.uv, 0.0, 1.0);
        // out.color = vec4<f32>(abs(chunk_pos.xz - world_data.player_pos.xz)/100.0, 0.0, 0.0);
        // out.color = vec4(chunk_pos.xz - mesh.world_position.xz, 0.0, 0.0);
        // return out;
    }
    let screen_uv = mesh.position.xy/vec2<f32>(world_data.screen_resolution.xy);
    let ray_origin = world_data.player_pos.xyz;
    let backface_hit_pos = mesh.world_position.xyz;
    let ray_dir = normalize(backface_hit_pos - ray_origin);
    let chunk_size = f32(side) * voxel_size / 2.0;
    let chunk_center = chunk_pos;

    // calculate ray hit pos
    
    var o = ray_origin;
    // how much t for 1 voxel unit in this direction
    var dt = (1.0 / ray_dir);

    // https://gamedev.stackexchange.com/a/18459
    // we know the ray hits, so we skip the 2 checks for ray not hitting the box
    let v1 = chunk_center - chunk_size;
    let v2 = chunk_center + chunk_size;
    let t1 = (v1 - o)*dt;
    let t2 = (v2 - o)*dt;
    let tmin = min(t1, t2);

    // if t is -ve, we want to march from the ray_origin instead. so clip it at 0.0
    var t = max(0.0, max(tmin.x, max(tmin.y, tmin.z)));

    o = ray_origin + ray_dir * t;

    // cache the hit position
    let ray_pos = o;


    // marching inside the chunk

    // we only want the magnitude of dt for marching
    dt = abs(dt);
    // origin wrt chunk's origin
    o -= (chunk_center - chunk_size);
    // make each voxel of size 1
    o /= voxel_size;

    // TODO: could try to rewrite this to make ray direction always be >= 0 in every direction
    // and store an (offset, sign) pair to calculate the voxel position
    // march * sign + offset
    // offset.x is 'side' and sign.x is -1 if ray goes towards -ve x for example.
    // after this, it becomes much easier to do the 2 component thing i think.

    var stepi = vec3<i32>(ray_dir >= 0.0) - vec3<i32>(ray_dir < 0.0);
    var step = vec3<f32>(stepi);
    var res: MipResult;
    res = inline_no_mip_loop(o, ray_dir, step, dt);
    res.t *= voxel_size;
    res.t += t;

    out.color = res.color;

    if res.hit {
        return out;
    }

    discard;
}

struct MipResult {
    color: vec4<f32>,
    hit: bool,
    t: f32,
}

fn inline_no_mip_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, dt: vec3<f32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * 0.001;

    // how much t untill we hit a plane along this axis
    var t = (step * 0.5 + 0.5 - fract(o) * step) * dt;
    // current voxel position (offset to center of the voxel)
    var march = floor(o) + 0.5;

    var voxel: u32;
    var mask: vec3<f32>;
    for (var i = 0u; i < side * 3u - 2u; i += 1u) {
        voxel = get_voxel(march);
        if (voxel != 0u) {
            res.color = get_color(voxel);
            res.hit = true;
            return res;
        }

        mask = vec3<f32>(t.xyz <= min(t.yzx, t.zxy));
        t += mask * dt;
        march += mask * step;
    }

    return res;
}

