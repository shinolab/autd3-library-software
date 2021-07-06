#version 450 core

in ivec3 a_pos;
out vec3 v_gpos;
uniform mat4 u_model_view_proj;
uniform mat4 u_model;

void main() {
    gl_Position = u_model_view_proj * vec4(a_pos, 1.0);
    v_gpos = vec3(u_model * vec4(a_pos, 1.0));
}
