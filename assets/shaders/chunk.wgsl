#import bevy_pbr::forward_io::VertexOutput

@group(1) @binding(0) var<uniform> side: u32;
@group(1) @binding(5) var<uniform> chunk_pos: vec3<f32>;
@group(1) @binding(6) var<uniform> chunk_size: f32;

// 4 voxels per u32 TODO: little endian or big endian ? :/ (does it even matter?)
@group(1) @binding(1) var voxels: texture_3d<u32>;
@group(1) @binding(2) var materials: texture_1d<f32>;

@group(1) @binding(3) var<uniform> pos: vec3<f32>;
@group(1) @binding(4) var<uniform> resolution: vec2<f32>;

// returns voxel material in the rightmost byte
fn get_voxel(pos: vec3<f32>) -> u32 {
    if (any(pos >= f32(side) || pos < 0.0)) {
        return 0u;
    }
    
    var coords = vec3<u32>(pos);
    let xyzw = coords.x % 4u;
    coords.x /= 4u;

    // loading textures automatically break the u32 into a vec4<u32> 1 byte each channel (~~ vec4<u8>)
    var voxel = textureLoad(voxels, coords, 0);

    return voxel[xyzw];
}

fn get_color(index: u32) -> vec4<f32> {
    return textureLoad(materials, index, 0);
}

// https://www.shadertoy.com/view/4dX3zl
// http://www.cse.yorku.ca/~amana/research/grid.pdf
// could also explore sphere/cube assisted marching
// - [Raymarching Voxel](https://medium.com/@calebleak/raymarching-voxel-rendering-58018201d9d6)
@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    let screen_uv = mesh.position.xy/resolution.xy;
    let ray_origin = pos.xyz;
    let ray_dir = normalize(mesh.world_position.xyz - ray_origin);
    let voxel_size = chunk_size * 2.0 / f32(side);
    let chunk_center = chunk_pos;

    // calculate ray hit pos
    
    var o = ray_origin;
    // how much t for 1 voxel unit in this direction
    var dt = (1.0 / ray_dir);

    // https://gamedev.stackexchange.com/a/18459
    // we know the ray hits, so we skip the 2 checks for ray not hitting the box
    let lowest = chunk_center - chunk_size;
    let highest = chunk_center + chunk_size;
    let t1 = (lowest - o)*dt;
    let t2 = (highest - o)*dt;
    let tmin = min(t1, t2);

    // if t is -ve, we want to march from the ray_origin instead. so clip it at 0.0
    o += ray_dir * max(0.0, max(tmin.x, max(tmin.y, tmin.z)));

    // cache the hit position
    let ray_pos = o;


    // marching inside the chunk

    // we only want the magnitude of dt for marching
    dt = abs(dt);
    // origin wrt chunk's origin
    o -= (chunk_pos - chunk_size);
    // make each voxel of size 1
    o /= voxel_size;

    var step = sign(ray_dir);
    // how much t untill we hit a plane along this axis
    var t = (step * 0.5 + 0.5 - fract(o) * step) * dt;
    // current voxel position (offset to center of the voxel)
    var march = floor(o) + 0.5;

    var voxel: u32;
    var mask: vec3<f32>;
    for (var i = 0u; i < side * 3u - 2u; i += 1u) {
        voxel = get_voxel(march);
        if (voxel != 0u) {
            break;
        }

        mask = vec3<f32>(t.xyz <= min(t.yzx, t.zxy));
        t += mask * dt;
        march += mask * step;
    }
    if (voxel == 0u) {
        discard;
    }
    // let masked_t = mask * t;
    // let final_t = max(masked_t.x, max(masked_t.y, masked_t.z)) * voxel_size;
    // let ray_hit_pos = ray_pos + final_t * ray_dir;

    var color = vec3<f32>(1.0);
    var alpha = 1.0;
    // color = vec3<f32>(20.0/length(ray_hit_pos - ray_origin));
    // color = vec3<f32>(20.0/length(march * voxel_size + (chunk_pos + chunk_size) - ray_origin));
    color = get_color(voxel).xyz;
    return vec4<f32>(color, alpha);
}
