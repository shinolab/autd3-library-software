using AUTD3Sharp;
using UnityEngine;

public class SimpleAUTDController : MonoBehaviour
{
    AUTD _autd = new AUTD();
    public GameObject Target;

    void Awake()
    {
        _autd = new AUTD();

        // 単位や座標系はUnity空間に合わせてあります
        // つまり, 単位はm, 座標系は"z"軸の反転した左手座標系です
        // また, 回転指定にクオータニオンを直接指定できます.
        _autd.AddDevice(gameObject.transform.position, gameObject.transform.rotation);

        _autd.Open();
        _autd.AppendModulationSync(AUTD.SineModulation(150)); // 150 Hz
    }

    void Update()
    {
        if (Target != null)
            _autd.AppendGainSync(AUTD.FocalPointGain(Target.transform.position));
    }

    private void OnApplicationQuit()
    {
        _autd.Dispose();
    }
}
