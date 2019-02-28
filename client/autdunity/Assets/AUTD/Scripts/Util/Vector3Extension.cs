using System.Collections;
using System.Collections.Generic;
using UnityEngine;

static class Vector3fExtension
{
    public static void Rectify(ref this Vector3 vec) 
    {
        vec.x = Mathf.Abs(vec.x);
        vec.y = Mathf.Abs(vec.y);
        vec.z = Mathf.Abs(vec.z);
    }
}
