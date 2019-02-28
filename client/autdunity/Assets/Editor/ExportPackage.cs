using UnityEditor;
using UnityEngine;

public class ExportPackage
{
    [MenuItem("Build/Export Package")]
    static void Export()
    {
        string[] file = {
            "Assets/AUTD/Scripts/AUTD3Sharp.cs",
            "Assets/AUTD/Scripts/NativeMethods.cs",
            "Assets/AUTD/Scripts/Util/GainMap.cs",
            "Assets/AUTD/Scripts/AUTD/Matrix3x3.cs",
            "Assets/AUTD/Scripts/AUTD/Vector3Extension.cs",
            "Assets/AUTD/Example/AUTD.prefab",
            "Assets/AUTD/Example/Simple.cs",
            "Assets/AUTD/Plugins/x86_64/AUTD3SharpDll.dll"
        };

        AssetDatabase.ExportPackage(file, "autd3.unitypackage", ExportPackageOptions.Recurse | ExportPackageOptions.Default);

        Debug.Log("Exported!");
    }
}
