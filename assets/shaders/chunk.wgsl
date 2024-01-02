#import bevy_pbr::forward_io::VertexOutput

@group(1) @binding(0) var<uniform> side: u32;
@group(1) @binding(5) var<uniform> _chunk_pos: vec3<f32>;
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

// https://www.shadertoy.com/view/4dX3zl
// http://www.cse.yorku.ca/~amana/research/grid.pdf
@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    let screen_uv = mesh.position.xy/resolution.xy;
    let ray_pos = mesh.world_position.xyz;
    let ray_origin = pos.xyz;
    let ray_dir = normalize(ray_pos - ray_origin);
    let voxel_size = chunk_size * 2.0 / f32(side);

    // chunk pos at it's center
    var chunk_pos = _chunk_pos;
    // chunk pos at the corner
    chunk_pos -= chunk_size;

    var o = ray_pos;
    // origin wrt chunk's origin
    o -= chunk_pos;
    // make each voxel of size 1
    o /= voxel_size;


    // how much t for 1 voxel unit in this direction
    var dt = abs(1.0 / ray_dir);
    var step = sign(ray_dir);
    // how much t untill we hit a plane along this axis
    var t = (step * 0.5 + 0.5 - fract(o) * step) * dt;
    // current voxel position (offset to center of the voxel)
    var march = floor(o) + 0.5;

    var voxel: u32;
    var mask: vec3<f32>;
    for (var i = 0u; i < side * 3u; i += 1u) {
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
    let masked_t = mask * t;
    let final_t = max(masked_t.x, max(masked_t.y, masked_t.z)) * voxel_size;
    let ray_hit_pos = ray_pos + final_t * ray_dir;

    var color = vec3<f32>(1.0);
    var alpha = 1.0;
    color = vec3<f32>(20.0/length(ray_hit_pos - ray_origin));
    // color = vec3<f32>(20.0/length(march * voxel_size + chunk_pos - ray_origin));
    return vec4<f32>(color, alpha);
}
