#version 450 core

in vec3 v_gpos;
out vec4 o_Color;

uniform float u_wavenum;
uniform float u_color_scale;
uniform float u_trans_size;
uniform float u_trans_num;
uniform sampler1D u_color_map;
uniform sampler1D u_trans_pos;
uniform sampler1D u_trans_pos_256;
uniform sampler1D u_trans_pos_sub;
uniform sampler1D u_trans_drive;

const float PI = 3.141592653589793;

vec4 coloring(float t)
{
  return texture(u_color_map, clamp(t * u_color_scale, 0.0, 1.0));
}

void main() {
    float re = 0.0;
    float im = 0.0;
    for(float idx = 0.0; idx < 65536.0; idx++){
        if (idx >= u_trans_num) break;
        vec3 t = texture(u_trans_pos, (idx+0.5) / u_trans_num).xyz;
        vec3 t_256 = texture(u_trans_pos_256, (idx+0.5) / u_trans_num).xyz;
        vec3 t_sub = texture(u_trans_pos_sub, (idx+0.5) / u_trans_num).xyz;
        vec3 tr = floor(255.0 * t);
        vec3 tr_256 = 256.0 * floor(255.0 * t_256);
        vec3 tp = u_trans_size * (tr + tr_256 + t_sub);
        float p = 2.0*PI*texture(u_trans_drive, (idx+0.5) / u_trans_num).x;
        float amp = texture(u_trans_drive, (idx+0.5) / u_trans_num).y;
        float d = length(v_gpos - tp);
        im += amp * cos(p - u_wavenum*d) / d;
        re += amp * sin(p - u_wavenum*d) / d;
    }
    float c = sqrt(re*re+im*im);
    o_Color = coloring(c);
}
