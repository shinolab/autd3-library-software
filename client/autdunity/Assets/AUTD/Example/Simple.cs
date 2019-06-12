using AUTD3Sharp;
using UnityEngine;

public class Simple : MonoBehaviour
{
    AUTD _autd = new AUTD();

    void Awake()
    {
        _autd = new AUTD();
        _autd.AddDevice(Vector3.zero, Vector3.zero);
        _autd.Open();

        _autd.AppendModulationSync(AUTD.SineModulation(150));
        _autd.AppendGainSync(AUTD.FocalPointGain(new Vector3(-80, 60, 150) * 0.001f));
    }

    private void OnApplicationQuit()
    {
        _autd.Dispose();
    }
}
