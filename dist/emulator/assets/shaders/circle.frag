#version 450 core

in vec2 v_TexCoord;
out vec4 o_Color;
uniform vec4 i_Color;
uniform sampler2D t_color;

void main() {
    vec4 tex = texture(t_color, v_TexCoord);
    o_Color = vec4(vec3(i_Color), tex.a);
}