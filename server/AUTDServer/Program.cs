using System;
using System.Collections.Generic;
using System.Xml;
using TCatSysManagerLib;
using System.Text;
using TwinCAT.SystemManager;
using EnvDTE;
using EnvDTE80;
using System.IO;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using TwinCAT.Ads;

namespace AUTDServer
{
    class Program
    {
        static String SOLUTION_NAME = "TwinCATAUTDServer";

        [STAThread] 
        static void Main(string[] args)
        {
            string SOLUTION_PATH = Path.Combine(
                Environment.GetEnvironmentVariable("temp"),
                SOLUTION_NAME);
            MessageFilter.Register();
            try
            {
                // Parse Commandline Options
                bool alwaysYes = false;
                string[] cmds = System.Environment.GetCommandLineArgs();
                foreach (string cmd in cmds)
                {
                    if (cmd == "-y") alwaysYes = true;
                }

                // Close all TwinCAT Autd Server solutions currently opened
                IEnumerable<System.Diagnostics.Process> processes = System.Diagnostics.Process.GetProcesses().Where(x => x.MainWindowTitle.StartsWith(SOLUTION_NAME) && x.ProcessName.Contains("devenv"));
                foreach (var process in processes)
                {
                    DTE cdte = GetDTE(process.Id);
                    if (cdte != null)
                    {
                        cdte.Quit();
                    }
                }

                // Wait for input
                Console.WriteLine("Please Enter the IP Address of your Client to allow connection: [127.0.0.1]");
                IPAddress ipaddr;
                String ipaddrStr = "127.0.0.1";
                if (!alwaysYes)
                    ipaddrStr = Console.ReadLine();

                IPAddress.TryParse(ipaddrStr, out ipaddr);

                Console.WriteLine("Connecting to TcXaeShell DTE...");
                Type t = System.Type.GetTypeFromProgID("TcXaeShell.DTE.15.0");
                EnvDTE80.DTE2 dte = (EnvDTE80.DTE2)System.Activator.CreateInstance(t);
            
                dte.SuppressUI = false;
                dte.MainWindow.Visible = true;
                dte.UserControl = true;


                Console.WriteLine("Switching TwinCAT3 to Config Mode...");
                SetConfigMode();
                System.Threading.Thread.Sleep(1000);
                Console.WriteLine("Creating a Project...");
                Project project = CreateProject(dte, SOLUTION_PATH);
                ITcSysManager sysManager = project.Object;
                if (ipaddr != null){
                    Console.WriteLine("Setting up the Routing Table to " + ipaddr.ToString());
                    AddRoute(sysManager, ipaddr);
                }
                Console.WriteLine("Scanning Devices...");
                List<ITcSmTreeItem> autds = ScanAUTDs(sysManager);
                AssignCPUCores(sysManager);
                SetupTask(sysManager, autds);
                Console.WriteLine("Activating and Restarting TwinCAT3...");
                sysManager.ActivateConfiguration();
                sysManager.StartRestartTwinCAT();
                Console.WriteLine("Saving the Project...");
                SaveProject(dte, project, SOLUTION_PATH);
                Console.WriteLine("Done. Do you want to close the TwinCAT config window? [Yes]/No");
                
                String closeWindow = "Yes";
                if (!alwaysYes)
                    closeWindow = Console.ReadLine();
                if (closeWindow != "No") dte.Quit();
            }
            catch (Exception e)
            {
                Console.WriteLine("Error. Press any key to exit. Check your license of TwinCAT3.");
                Console.WriteLine(e.Message);
                Console.ReadLine();

            }

            MessageFilter.Revoke();
 
        }

        [DllImport("ole32.dll")]
        private static extern int CreateBindCtx(uint reserved, out IBindCtx ppbc);

        public static DTE GetDTE(int processId)
        {
            string progId = "!TcXaeShell.DTE.15.0:" + processId.ToString();
            object runningObject = null;

            IBindCtx bindCtx = null;
            IRunningObjectTable rot = null;
            IEnumMoniker enumMonikers = null;

            try
            {
                Marshal.ThrowExceptionForHR(CreateBindCtx(reserved: 0, ppbc: out bindCtx));
                bindCtx.GetRunningObjectTable(out rot);
                rot.EnumRunning(out enumMonikers);

                IMoniker[] moniker = new IMoniker[1];
                IntPtr numberFetched = IntPtr.Zero;
                while (enumMonikers.Next(1, moniker, numberFetched) == 0)
                {
                    IMoniker runningObjectMoniker = moniker[0];

                    string name = null;

                    try
                    {
                        if (runningObjectMoniker != null)
                        {
                            runningObjectMoniker.GetDisplayName(bindCtx, null, out name);
                        }
                    }
                    catch (UnauthorizedAccessException)
                    {
                        // Do nothing, there is something in the ROT that we do not have access to.
                    }

                    if (!string.IsNullOrEmpty(name) && string.Equals(name, progId, StringComparison.Ordinal))
                    {
                        Marshal.ThrowExceptionForHR(rot.GetObject(runningObjectMoniker, out runningObject));
                        break;
                    }
                }
            }
            finally
            {
                if (enumMonikers != null)
                {
                    Marshal.ReleaseComObject(enumMonikers);
                }

                if (rot != null)
                {
                    Marshal.ReleaseComObject(rot);
                }

                if (bindCtx != null)
                {
                    Marshal.ReleaseComObject(bindCtx);
                }
            }

            return (DTE)runningObject;
        } 

        static void SetConfigMode()
        {
            TcAdsClient client = new TcAdsClient();
            StateInfo mode = new StateInfo();

            client.Connect((int)AmsPort.SystemService);
            mode.AdsState = client.ReadState().AdsState;
            mode.AdsState = AdsState.Reconfig;
            client.WriteControl(mode);
            client.Dispose();
        }

        static void DeleteDirectory(string path)
        {
            foreach (string directory in Directory.GetDirectories(path))
            {
                DeleteDirectory(directory);
            }

            try
            {
                Directory.Delete(path, true);
            }
            catch (IOException)
            {
                Directory.Delete(path, true);
            }
            catch (UnauthorizedAccessException)
            {
                Directory.Delete(path, true);
            }
        }

        static Project CreateProject(EnvDTE80.DTE2 dte, string path)
        {
            if (Directory.Exists(path))
                DeleteDirectory(path);
            Directory.CreateDirectory(path);
            //Directory.CreateDirectory(@"C:\Temp\SolutionFolder\MySolution1");

            Solution2 solution = (Solution2)dte.Solution;
            solution.Create(path, SOLUTION_NAME);
            solution.SaveAs(Path.Combine(path, SOLUTION_NAME + ".sln"));

            string template = @"C:\TwinCAT\3.1\Components\Base\PrjTemplate\TwinCAT Project.tsproj"; //path to project template
            return solution.AddFromTemplate(template, path, SOLUTION_NAME);
        }

        static void SaveProject(EnvDTE80.DTE2 dte, Project project, string path)
        {
            project.Save();
            dte.Solution.SaveAs(Path.Combine(path, SOLUTION_NAME + ".sln"));
            Console.WriteLine("The Solution was saved at "+path+".");
        }

        static void AddRoute(ITcSysManager sysManager, IPAddress ipaddr)
        {
            ITcSmTreeItem routeConfiguration = sysManager.LookupTreeItem("TIRR");
            string addProjectRouteIp = @"<TreeItem>
                                           <RoutePrj>
                                             <AddProjectRoute>
                                               <Name>" + ipaddr.ToString() + @"</Name>
                                               <NetId>" + ipaddr.ToString() + @".1.1" + @"</NetId>
                                               <IpAddr>" + ipaddr.ToString() + @"</IpAddr>
                                             </AddProjectRoute>
                                           </RoutePrj>
                                         </TreeItem>";

            routeConfiguration.ConsumeXml(addProjectRouteIp);
        }

        static List<ITcSmTreeItem> ScanAUTDs(ITcSysManager sysManager)
        {
            ITcSmTreeItem3 devices = (ITcSmTreeItem3)sysManager.LookupTreeItem("TIID");
            XmlDocument doc = new XmlDocument();
            string xml = devices.ProduceXml(false);
            doc.LoadXml(xml);
            XmlNodeList nodes = doc.SelectNodes("TreeItem/DeviceGrpDef/FoundDevices/Device");
            List<XmlNode> ethernetNodes = new List<XmlNode>();
            foreach (XmlNode node in nodes)
            {
                XmlNode typeNode = node.SelectSingleNode("ItemSubType");

                int subType = int.Parse(typeNode.InnerText);

#pragma warning disable 0618
                if (subType == (int)DeviceType.EtherCAT_AutomationProtocol || subType == (int)DeviceType.Ethernet_RTEthernet || subType == (int)DeviceType.EtherCAT_DirectMode || subType == (int)DeviceType.EtherCAT_DirectModeV210)
#pragma warning restore 0618
                {
                    ethernetNodes.Add(node);
                }
            }

            if (ethernetNodes.Count == 0)
            {
                throw new Exception("No devices were found. Check if TwinCAT3 is in Config Mode");
            }

            Console.WriteLine(string.Format("Scan found a RT-compatible Ethernet device.", ethernetNodes.Count));
            ITcSmTreeItem3 device = (ITcSmTreeItem3)devices.CreateChild("EtherCAT Master", (int)DeviceType.EtherCAT_DirectMode, null, null);

            // Taking only the first found Ethernet Adapter
            XmlNode ethernetNode = ethernetNodes[0];
            XmlNode addressInfoNode = ethernetNode.SelectSingleNode("AddressInfo");
            addressInfoNode.SelectSingleNode("Pnp/DeviceDesc").InnerText = "TwincatEthernetDevice";
            // Set the Address Info
            string xml2 = string.Format("<TreeItem><DeviceDef>{0}</DeviceDef></TreeItem>", addressInfoNode.OuterXml);
            device.ConsumeXml(xml2);

            string scanxml = "<TreeItem><DeviceDef><ScanBoxes>1</ScanBoxes></DeviceDef></TreeItem>";
            device.ConsumeXml(scanxml);
            List<ITcSmTreeItem> autds = new List<ITcSmTreeItem>();
            foreach (ITcSmTreeItem box in device)
            {
                if (box.ItemSubTypeName == "AUTD")
                {
                    
                    XmlDocument bdoc = new XmlDocument();
                    string bxml = box.ProduceXml(false);
                    bdoc.LoadXml(bxml);

                    // set DC
                    XmlNodeList dcOpmodes = bdoc.SelectNodes("TreeItem/EtherCAT/Slave/DC/OpMode");
                    foreach (XmlNode item in dcOpmodes)
                    {
                        if (item.SelectSingleNode("Name").InnerText == "DC")
                        {
                            XmlAttribute attr = bdoc.CreateAttribute("Selected");
                            attr.Value = "true";
                            item.Attributes.SetNamedItem(attr);
                        }
                        else
                        {
                            item.Attributes.RemoveAll();
                        }
                    }
                    box.ConsumeXml(bdoc.OuterXml);
                     

                    autds.Add(box);
                }
            }

            Console.WriteLine(string.Format("{0} AUTDs are found and added.", autds.Count));

            return autds;
        }

        static void SetupTask(ITcSysManager sysManager, List<ITcSmTreeItem> autds)
        {
            const int HEAD_SIZE = 64;
            const int BODY_SIZE = 249;

            ITcSmTreeItem tasks = sysManager.LookupTreeItem("TIRT");
            ITcSmTreeItem task1 = tasks.CreateChild("Task 1", 0, null, null);
            XmlDocument doc = new XmlDocument();
            string xml = task1.ProduceXml(false);
            doc.LoadXml(xml);

            // set cycle: 1ms
            doc.SelectSingleNode("TreeItem/TaskDef/CycleTime").InnerText = "5000";
            task1.ConsumeXml(doc.OuterXml);

            ITcSmTreeItem task1out = sysManager.LookupTreeItem("TIRT^Task 1^Outputs");
            // make global header
            for (int i = 0; i < HEAD_SIZE; i++)
            {
                string name = string.Format("header[{0}]", i);
                task1out.CreateChild(name, -1, null, "WORD");
            }
            // make gain body
            for (int id = 0; id < autds.Count; id++)
            {
                for (int i = 0; i < BODY_SIZE; i++)
                {
                    string name = string.Format("gbody[{0}][{1}]", id, i);
                    task1out.CreateChild(name, -1, null, "WORD");
                }
            }
            // connect links
            for (int id = 0; id < autds.Count; id++)
            {
                for (int i = 0; i < HEAD_SIZE; i++)
                {
                    string source = string.Format("TIRT^Task 1^Outputs^header[{0}]", i);
                    string destination = string.Format("TIID^EtherCAT Master^Box {0} (AUTD)^RxPdo1^data[{1}]", id+1, i);
                    sysManager.LinkVariables(source, destination);
                }
                for (int i = 0; i < BODY_SIZE; i++)
                {
                    string source = string.Format("TIRT^Task 1^Outputs^gbody[{0}][{1}]", id, i);
                    string destination = string.Format("TIID^EtherCAT Master^Box {0} (AUTD)^RxPdo0^data[{1}]", id+1, i);
                    sysManager.LinkVariables(source, destination);
                }
            }

        }

        [Flags()]
        public enum CpuAffinity : ulong
        {
            CPU1 = 0x0000000000000001,
            CPU2 = 0x0000000000000002,
            CPU3 = 0x0000000000000004,
            CPU4 = 0x0000000000000008,
            CPU5 = 0x0000000000000010,
            CPU6 = 0x0000000000000020,
            CPU7 = 0x0000000000000040,
            CPU8 = 0x0000000000000080,
            None = 0x0000000000000000,
            MaskSingle = CPU1,
            MaskDual = CPU1 | CPU2,
            MaskQuad = MaskDual | CPU3 | CPU4,
            MaskHexa = MaskQuad | CPU5 | CPU6,
            MaskOct = MaskHexa | CPU7 | CPU8,
            MaskAll = 0xFFFFFFFFFFFFFFFF
        }
        static public void AssignCPUCores(ITcSysManager sysManager)
        {
            ITcSmTreeItem realtimeSettings = sysManager.LookupTreeItem("TIRS");
            // CPU Settings
            // <TreeItem>
            // <RTimeSetDef>
            // <MaxCPUs>3</MaxCPUs>
            // <Affinity>#x0000000000000007</Affinity>
            // <CPUs>
            // <CPU id="0">
            // <LoadLimit>10</LoadLimit>
            // <BaseTime>10000</BaseTime>
            // <LatencyWarning>200</LatencyWarning>
            // </CPU>
            // </CPUs>
            // </RTimeSetDef>
            // </TreeItem> 
            string xml = null;
            MemoryStream stream = new MemoryStream();
            StringWriter stringWriter = new StringWriter();
            using (XmlWriter writer = XmlTextWriter.Create(stringWriter))
            {
                writer.WriteStartElement("TreeItem");
                writer.WriteStartElement("RTimeSetDef");
                writer.WriteElementString("MaxCPUs", "1");
                //string affinityString = string.Format("#x{0}", ((ulong)
                //cpuAffinity.MaskQuad).ToString("x16"));
                //writer.WriteElementString("Affinity", affinityString);
                writer.WriteStartElement("CPUs");
                writeCpuProperties(writer, 0, 5000);
                writer.WriteEndElement();     // CPUs     
                writer.WriteEndElement();     // RTimeSetDef     
                writer.WriteEndElement();     // TreeItem
            }
            xml = stringWriter.ToString();
            realtimeSettings.ConsumeXml(xml);
        }

        static private void writeCpuProperties(XmlWriter writer, int id, /*int loadLimit, */int baseTime/*, int latencyWarning*/)
        {
            writer.WriteStartElement("CPU");
            writer.WriteAttributeString("id", id.ToString());
            //writer.WriteElementString("LoadLimit", loadLimit.ToString());
            writer.WriteElementString("BaseTime", baseTime.ToString());
            //writer.WriteElementString("LatencyWarning", latencyWarning.ToString());
            writer.WriteEndElement();
        }
    }


    public class MessageFilter : IOleMessageFilter
    {
        //
        // Class containing the IOleMessageFilter
        // thread error-handling functions.

        // Start the filter.
        public static void Register()
        {
            IOleMessageFilter newFilter = new MessageFilter();
            IOleMessageFilter oldFilter = null;
            CoRegisterMessageFilter(newFilter, out oldFilter);
        }

        // Done with the filter, close it.
        public static void Revoke()
        {
            IOleMessageFilter oldFilter = null;
            CoRegisterMessageFilter(null, out oldFilter);
        }

        //
        // IOleMessageFilter functions.
        // Handle incoming thread requests.
        int IOleMessageFilter.HandleInComingCall(int dwCallType,
          System.IntPtr hTaskCaller, int dwTickCount, System.IntPtr
          lpInterfaceInfo)
        {
            //Return the flag SERVERCALL_ISHANDLED.
            return 0;
        }

        // Thread call was rejected, so try again.
        int IOleMessageFilter.RetryRejectedCall(System.IntPtr
          hTaskCallee, int dwTickCount, int dwRejectType)
        {
            if (dwRejectType == 2)
            // flag = SERVERCALL_RETRYLATER.
            {
                // Retry the thread call immediately if return >=0 & 
                // <100.
                return 99;
            }
            // Too busy; cancel call.
            return -1;
        }

        int IOleMessageFilter.MessagePending(System.IntPtr hTaskCallee,
          int dwTickCount, int dwPendingType)
        {
            //Return the flag PENDINGMSG_WAITDEFPROCESS.
            return 2;
        }

        // Implement the IOleMessageFilter interface.
        [DllImport("Ole32.dll")]
        private static extern int
          CoRegisterMessageFilter(IOleMessageFilter newFilter, out 
          IOleMessageFilter oldFilter);
    }

    [ComImport(), Guid("00000016-0000-0000-C000-000000000046"),
    InterfaceTypeAttribute(ComInterfaceType.InterfaceIsIUnknown)]
    interface IOleMessageFilter
    {
        [PreserveSig]
        int HandleInComingCall(
            int dwCallType,
            IntPtr hTaskCaller,
            int dwTickCount,
            IntPtr lpInterfaceInfo);

        [PreserveSig]
        int RetryRejectedCall(
            IntPtr hTaskCallee,
            int dwTickCount,
            int dwRejectType);

        [PreserveSig]
        int MessagePending(
            IntPtr hTaskCallee,
            int dwTickCount,
            int dwPendingType);
    }
}
