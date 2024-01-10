#import bevy_pbr::forward_io::VertexOutput

@group(1) @binding(0) var<uniform> side: u32;
@group(1) @binding(5) var<uniform> chunk_pos: vec3<f32>;
@group(1) @binding(6) var<uniform> chunk_size: f32;

@group(1) @binding(1) var voxels: texture_3d<u32>;
@group(1) @binding(7) var voxels_mip1: texture_3d<u32>;
@group(1) @binding(8) var voxels_mip2: texture_3d<u32>;
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


fn get_mip1(pos: vec3<f32>) -> vec2<u32> {
    if (any(pos >= f32(side) || pos < 0.0)) {
        return vec2<u32>(0u);
    }

    var coords = vec3<u32>(pos);
    var voxel = textureLoad(voxels_mip1, coords/4u, 0);
    // return vec4<u32>(voxel.x & 31u, voxel.x >> 16u, voxel.y & 31u, voxel.y >> 16u);
    return voxel.xy;
}
fn get_mip2(pos: vec3<f32>) -> vec2<u32> {
    if (any(pos >= f32(side) || pos < 0.0)) {
        return vec2<u32>(0u);
    }

    var coords = vec3<u32>(pos);
    var voxel = textureLoad(voxels_mip2, coords/(4u * 8u), 0);
    return voxel.xy;
}

fn unpackBytes(t: u32) -> vec4<u32> {
    return (t >> vec4<u32>(0u, 8u, 16u, 24u)) & 15u;
}

struct Mip {
    z0: vec4<u32>,
    z1: vec4<u32>,
}

fn getMipByte(mip: vec2<u32>, index: u32) -> u32 {
    // if (index == 5u) {
    //     return 1u;
    // } else {
    //     return 0u;
    // }
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

// fn unpackMipBytes(mip: vec2<u32>, level: u32) -> array<u32, 8> {
//     switch (level) {
//         case 0u { // v1<u8> -> v8<u1>
//             let b1 = (mip.x >> vec4<u32>(0u, 1u, 2u, 3u)) & 1u;
//             let b2 = (mip.x >> vec4<u32>(4u, 5u, 6u, 7u)) & 1u;
//             let b1 = (mip.x >> vec4<u32>(0u, 1u, 2u, 3u)) & 1u;
//             let b2 = (mip.x >> vec4<u32>(4u, 5u, 6u, 7u)) & 1u;
//             return array(b1.x, b1.y, b1.z, b1.w, b2.x, b2.y, b2.z, b2.w);
//         }
//         case 1u { // v2<
//             let b1 = (box >> vec4<u32>(0u, 1u, 2u, 3u) * 8u) & 15u;
//             let b2 = (box >> vec4<u32>(4u, 5u, 6u, 7u)) & 15u;
//             return array(b1.x, b1.y, b1.z, b1.w, b2.x, b2.y, b2.z, b2.w);
//         }
//         case 2u {
//             // return mat2x4(0u);
//         }
//         default {
//             // return mat2x4(0u);
//         }
//     }
//     return array(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
// }

fn get_bytemasks() -> array<u32, 4> {return array(15u, 15u << 8u, 15u << 16u, 15u << 24u);}
fn get_bitmasks() -> array<u32, 8> {
    return array(
        1u | 1u << 8u | 1u << 16u | 1u << 24u,
        2u | 2u << 8u | 2u << 16u | 2u << 24u,
        4u | 4u << 8u | 4u << 16u | 4u << 24u,
        8u | 8u << 8u | 8u << 16u | 8u << 24u,
        16u | 16u << 8u | 16u << 16u | 16u << 24u,
        32u | 32u << 8u | 32u << 16u | 32u << 24u,
        64u | 64u << 8u | 64u << 16u | 64u << 24u,
        128u | 128u << 8u | 128u << 16u | 128u << 24u,
    );
}

fn mip0(march: vec3<f32>) -> bool {
    var mip = get_mip1(march);
    var marchu = vec3<i32>(march);
    var mod4 = marchu % 4;
    var m = vec3<u32>(mod4 > 1) << vec3(0u, 1u, 2u);
    var index = m.x | m.y | m.z;
    let comp = getMipByte(mip, index);
    var mod2 = marchu % 2;
    m = vec3<u32>(mod2 > 0) << vec3(0u, 1u, 2u);
    index = m.x | m.y | m.z;
    let voxel = getMipBit(comp, index);
    return voxel != 0u;
}
fn mip1(march: vec3<f32>) -> bool {
    var mip = get_mip1(march);
    var marchu = vec3<i32>(march);
    var mod4 = marchu % 4;
    var m = vec3<u32>(mod4 > 1) << vec3(0u, 1u, 2u);
    var index = m.x | m.y | m.z;
    let comp = getMipByte(mip, index);
    return comp != 0u;
}
fn mip2(march: vec3<f32>) -> bool {
    var mip = get_mip1(march);
    return any(mip > 0u);
}
fn mip3(march: vec3<f32>) -> bool {
    var mip = get_mip2(march);
    var marchu = vec3<i32>(march);
    var mod4 = marchu % 32;
    var m = vec3<u32>(mod4 > 15) << vec3(0u, 1u, 2u);
    var index = m.x | m.y | m.z;
    let comp = getMipByte(mip, index);
    var mod2 = marchu % 16;
    m = vec3<u32>(mod2 > 7) << vec3(0u, 1u, 2u);
    index = m.x | m.y | m.z;
    let voxel = getMipBit(comp, index);
    return voxel != 0u;
}
fn mip4(march: vec3<f32>) -> bool {
    var mip = get_mip2(march);
    var marchu = vec3<i32>(march);
    var mod4 = marchu % 32;
    var m = vec3<u32>(mod4 > 15) << vec3(0u, 1u, 2u);
    var index = m.x | m.y | m.z;
    let comp = getMipByte(mip, index);
    return comp != 0u;
}
fn mip5(march: vec3<f32>) -> bool {
    var mip = get_mip2(march);
    return any(mip > 0u);
}

// https://www.shadertoy.com/view/4dX3zl
// http://www.cse.yorku.ca/~amana/research/grid.pdf
// could also explore sphere/cube assisted marching
// - [Raymarching Voxel](https://medium.com/@calebleak/raymarching-voxel-rendering-58018201d9d6)
// - [VertexOutput](https://github.com/bevyengine/bevy/blob/71adb77a2ea97027ae54dea5552ba9fdbfb707fd/crates/bevy_pbr/src/render/forward_io.wgsl#L30)
@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    if (1.0 > 0.0) {
        // let v1 = get_mip1(vec3<f32>(0.01));
        // let v2 = get_mip2(vec3<f32>(0.01));
        // return vec4<f32>(f32(v1.x + v2.x));
        // return vec4<f32>(1.0);
        // return vec4<f32>((mesh.world_position.xyz - chunk_pos)/1000.0, 1.0);
        // return vec4<f32>(mesh.uv, 1.0, 1.0);
    }
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
    // dt = normalize(dt);
    // origin wrt chunk's origin
    o -= (chunk_pos - chunk_size);
    // make each voxel of size 1
    o /= voxel_size;

    var step = sign(ray_dir);
    var stepi = vec3<i32>(step);
    var res = mip5_loop_final(o, ray_dir, step, stepi, dt);
    // var res = mip2_loop_final(o, ray_dir, step, stepi, dt);
    // var res = inline_mip2_loop(o, ray_dir, step, stepi, dt);
    // var res = inline_no_mip_loop(o, ray_dir, step, dt);

    if res.hit {
        return res.color;
    }

    discard;
}

struct MipResult {
    color: vec4<f32>,
    hit: bool,
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

fn inline_mip2_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * 0.001;
    // current voxel position (offset to center of the voxel)
    var march = floor(o) + 0.5;
    var marchi = vec3<i32>(march);

    // how much t untill we hit a plane along this axis
    var t = (step * (0.5 - fract(o/4.0)) + 0.5) * dt * 4.0;
    var voxel: u32;
    var mask: vec3<f32>;
    var last_t = 0.0;
    for (var i = 0u; i < (side / 4u) * 3u - 2u; i += 1u) {
        if any(marchi > i32(side) || marchi < 0) {
            break;
        }
        var mip = get_mip1(march);
        if any(mip > 0u) {
            var o = o + ray_dir * (last_t + 0.001);
            var march = floor(o) + 0.5;
            var marchi = vec3<i32>(march);
            var t = (step * (0.5 - fract(o/2.0)) + 0.5) * dt * 2.0;
            var mod4 = marchi % 4;
            var last_t = 0.0;
            for (var i2 = 0u; i2 < 4u; i2 += 1u) {
                if (any(mod4 >= 4 || mod4 < 0)) {
                    break;
                }
                let m = vec3<u32>(mod4 > 1) << vec3(0u, 1u, 2u);
                let index = m.x | m.y | m.z;
                let comp = getMipByte(mip, index);
                if (comp > 0u) {
                    var o = o + ray_dir * (last_t + 0.001);
                    var march = floor(o) + 0.5;
                    var marchi = vec3<i32>(march);
                    var t = (step * (0.5 - fract(o)) + 0.5) * dt;
                    var mod2 = marchi % 2;
                    var last_t = 0.0;
                    for (var i3 = 0u; i3 < 4u; i3 += 1u) {
                        if any(mod2 >= 2 || mod2 < 0) {
                            break;
                        }
                        let m = vec3<u32>(mod2 > 0) << vec3(0u, 1u, 2u);
                        let index = m.x | m.y | m.z;
                        let voxel = getMipBit(comp, index);
                        if voxel > 0u {
                            let voxel = get_voxel(march);
                            // check again cux of precision issues ig :/
                            if voxel > 0u {
                                res.color = get_color(voxel);
                                res.hit = true;
                                return res;
                            }
                        }
                        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
                        let mask = vec3<f32>(maski);
                        last_t = min(t.x, min(t.y, t.z));
                        t += mask * dt;
                        march += mask * step;
                        mod2 += maski * stepi;
                    }
                }
                let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
                let mask = vec3<f32>(maski);
                last_t = min(t.x, min(t.y, t.z));
                t += mask * dt * 2.0;
                march += mask * step * 2.0;
                mod4 += maski * stepi * 2;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt * 4.0;
        march += mask * step * 4.0;
        marchi += maski * stepi * 4;
    }

    return res;
}

fn mip5_loop_final(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * 0.001;
    // current voxel position (offset to center of the voxel)
    var march = floor(o) + 0.5;
    var marchi = vec3<i32>(march);

    // how much t untill we hit a plane along this axis
    var t = (step * (0.5 - fract(o/32.0)) + 0.5) * dt * 32.0;
    var last_t = 0.0;
    for (var i = 0u; i < (side / 32u) * 3u - 2u; i += 1u) {
        if any(marchi > i32(side) || marchi < 0) {
            break;
        }
        var mip = get_mip2(march);
        if any(mip > 0u) {
            let res = mip4_loop(o, ray_dir, step, stepi, dt, last_t, mip);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt * 32.0;
        march += mask * step * 32.0;
        marchi += maski * stepi * 32;
    }
    
    return res;
}

fn mip4_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, mip: vec2<u32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + 0.001);
    var march = floor(o) + 0.5;
    var marchi = vec3<i32>(march);
    var t = (step * (0.5 - fract(o/16.0)) + 0.5) * dt * 16.0;
    var mod32 = marchi % 32;
    var last_t = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        if (any(mod32 >= 32 || mod32 < 0)) {
            break;
        }
        let m = vec3<u32>(mod32 > 15) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let comp = getMipByte(mip, index);
        if (comp > 0u) {
            res = mip3_loop(o, ray_dir, step, stepi, dt, last_t, comp);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt * 16.0;
        march += mask * step * 16.0;
        mod32 += maski * stepi * 16;
    }
    
    return res;
}

fn mip3_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, comp: u32) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + 0.001);
    var march = floor(o) + 0.5;
    var marchi = vec3<i32>(march);
    var t = (step * (0.5 - fract(o/8.0)) + 0.5) * dt * 8.0;
    var mod16 = marchi % 16;
    var last_t = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        if any(mod16 >= 16 || mod16 < 0) {
            break;
        }
        let m = vec3<u32>(mod16 > 7) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let voxel = getMipBit(comp, index);
        if voxel > 0u {
            let res = mip2_loop(o, ray_dir, step, stepi, dt, last_t);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt * 8.0;
        march += mask * step * 8.0;
        mod16 += maski * stepi * 8;
    }

    return res;
}

fn mip2_loop_final(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * 0.001;
    // current voxel position (offset to center of the voxel)
    var march = floor(o) + 0.5;
    var marchi = vec3<i32>(march);

    // how much t untill we hit a plane along this axis
    var t = (step * (0.5 - fract(o/4.0)) + 0.5) * dt * 4.0;
    var last_t = 0.0;
    for (var i = 0u; i < (side / 4u) * 3u - 2u; i += 1u) {
        if any(marchi > i32(side) || marchi < 0) {
            break;
        }
        var mip = get_mip1(march);
        if any(mip > 0u) {
            let res = mip1_loop(o, ray_dir, step, stepi, dt, last_t, mip);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt * 4.0;
        march += mask * step * 4.0;
        marchi += maski * stepi * 4;
    }
    
    return res;
}

fn mip2_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + 0.001);
    var march = floor(o) + 0.5;
    var marchi = vec3<i32>(march);
    var t = (step * (0.5 - fract(o/4.0)) + 0.5) * dt * 4.0;
    var mod8 = marchi % 8;
    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if (any(mod8 >= 8 || mod8 < 0)) {
            break;
        }
        var mip = get_mip1(march);
        if any(mip > 0u) {
            let res = mip1_loop(o, ray_dir, step, stepi, dt, last_t, mip);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt * 4.0;
        march += mask * step * 4.0;
        marchi += maski * stepi * 4;
        mod8 += maski * stepi * 4;
    }
    
    return res;
}

fn mip1_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, mip: vec2<u32>) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + 0.001);
    var march = floor(o) + 0.5;
    var marchi = vec3<i32>(march);
    var t = (step * (0.5 - fract(o/2.0)) + 0.5) * dt * 2.0;
    var mod4 = marchi % 4;
    var last_t = 0.0;
    for (var i = 0; i < 4; i += 1) {
        if (any(mod4 >= 4 || mod4 < 0)) {
            break;
        }
        let m = vec3<u32>(mod4 > 1) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let comp = getMipByte(mip, index);
        if (comp > 0u) {
            res = mip0_loop(o, ray_dir, step, stepi, dt, last_t, comp);
            if res.hit {
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt * 2.0;
        march += mask * step * 2.0;
        mod4 += maski * stepi * 2;
    }
    
    return res;
}

fn mip0_loop(_o: vec3<f32>, ray_dir: vec3<f32>, step: vec3<f32>, stepi: vec3<i32>, dt: vec3<f32>, _last_t: f32, comp: u32) -> MipResult {
    var res: MipResult;
    res.hit = false;
    res.color = vec4(0.0);

    var o = _o + ray_dir * (_last_t + 0.001);
    var march = floor(o) + 0.5;
    var marchi = vec3<i32>(march);
    var t = (step * (0.5 - fract(o)) + 0.5) * dt;
    var mod2 = marchi % 2;
    var last_t = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        if any(mod2 >= 2 || mod2 < 0) {
            break;
        }
        let m = vec3<u32>(mod2 > 0) << vec3(0u, 1u, 2u);
        let index = m.x | m.y | m.z;
        let voxel = getMipBit(comp, index);
        if voxel > 0u {
            let voxel = get_voxel(march);
            // check again cux of precision issues ig :/
            if voxel > 0u {
                // return get_color(voxel) * vec4(vec3(0.5), 1.0);
                res.color = get_color(voxel);
                res.hit = true;
                return res;
            }
        }
        let maski = vec3<i32>(t.xyz <= min(t.yzx, t.zxy));
        let mask = vec3<f32>(maski);
        last_t = min(t.x, min(t.y, t.z));
        t += mask * dt;
        march += mask * step;
        mod2 += maski * stepi;
    }

    return res;
}
