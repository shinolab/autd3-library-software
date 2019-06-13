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
            "Assets/AUTD/Scripts/Util/Matrix3x3.cs",
            "Assets/AUTD/Example/AUTD.prefab",
            "Assets/AUTD/Example/SimpleAUTDController.cs",
            "Assets/AUTD/Plugins/x86_64/autd3capi.dll",
            "Assets/AUTD/Material/AUTD.mat",
            "Assets/AUTD/Texture/AUTD.png",
            "Assets/Scenes/simple.unity"
        };

        AssetDatabase.ExportPackage(file, "autd3.unitypackage", ExportPackageOptions.Recurse | ExportPackageOptions.Default);

        Debug.Log("Exported!");
    }
}
