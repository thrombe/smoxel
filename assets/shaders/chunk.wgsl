#import bevy_pbr::forward_io::VertexOutput

@group(1) @binding(0) var<uniform> side: u32;
@group(1) @binding(5) var<uniform> _chunk_pos: vec3<f32>;
@group(1) @binding(6) var<uniform> chunk_size: f32;

// 4 voxels per u32 TODO: little endian or big endian ? :/ (does it even matter?)
@group(1) @binding(1) var voxels: texture_3d<u32>;
@group(1) @binding(2) var materials: texture_1d<f32>; // idk how these textures work :(

@group(1) @binding(3) var<uniform> pos: vec3<f32>;
@group(1) @binding(4) var<uniform> resolution: vec2<f32>;

// returns voxel material in the rightmost byte
fn get_voxel(pos: vec3<f32>) -> u32 {
    var coords = vec3<u32>(pos);
    let xyzw = coords.x % 4u;
    coords.x /= 4u;
    // loading textures automatically break the u32 into a vec4<u32> 1 byte each channel (~~ vec4<u8>)
    var voxel = textureLoad(voxels, coords, 0);

    var ans: u32;
    switch (xyzw) {
        case 0u {
            ans = voxel.x;
        }
        case 1u {
            ans = voxel.y;
        }
        case 2u {
            ans = voxel.z;
        }
        case 3u {
            ans = voxel.w;
        }
        default {
            discard;
        }
    }
    return ans;
}

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
    // nudge nudge for edge conditions
    o -= 0.00001;

    var march = vec3<f32>(floor(o));
    var dt = abs(1.0 / ray_dir);
    var step = vec3<f32>(sign(ray_dir));
    var t = step * (march - o) + ((step * 0.5) + 0.5) * dt;
    // var mask: vec3<bool>;
    var voxel: u32;
    for (var i=0u; i<side*2u; i = i+1u) {
        voxel = get_voxel(march);
        if (voxel != 0u) {
            continue;
        }

        if (t.x < t.y) {
            if (t.x < t.y) {
                t.x += dt.x;
                march.x += step.x;
            } else {
                t.z += dt.z;
                march.z += step.z;
            }
        } else {
            if (t.y < t.z) {
                t.y += dt.y;
                march.y += step.y;
            } else {
                t.z += dt.z;
                march.z += step.z;
            }
        }
    }
    if (voxel == 0u) {
        discard;
    }

    var color = vec3<f32>(1.0);
    var alpha = 1.0;
    color = 0.7*vec3<f32>(1.0 - max(max(t.x, t.y), t.z)/f32(2u * side));
    return vec4<f32>(color, alpha);
}
