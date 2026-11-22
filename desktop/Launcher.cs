using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;
using System.Reflection;

[assembly: AssemblyTitle("PureBlur Studio")]
[assembly: AssemblyDescription("Offline video person blur")]
[assembly: AssemblyVersion("0.2.0.0")]

internal static class Launcher
{
    [STAThread]
    private static int Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string venv = Path.Combine(root, ".venv");
        string dependencies = Path.Combine(root, ".local-runtime");
        string python = Path.Combine(venv, "Scripts", "pythonw.exe");
        bool useVenv = File.Exists(python) && Directory.Exists(Path.Combine(venv, "Lib", "site-packages", "av"));
        if (!useVenv)
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            python = Path.Combine(home, @".cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe");
        }

        if (!File.Exists(python) || (!useVenv && !Directory.Exists(Path.Combine(dependencies, "av"))))
            return Fail("PureBlur's local runtime is missing. Keep this app in its original folder.");
        if (!File.Exists(Path.Combine(root, "pureblur", "gui.py")))
            return Fail("Move this icon back into the PureBlur Studio folder and open it again.");

        try
        {
            var info = new ProcessStartInfo(python, "-B -m pureblur.gui");
            info.WorkingDirectory = root;
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            info.EnvironmentVariables.Remove("PYTHONHOME");
            info.EnvironmentVariables.Remove("PYTHONPATH");
            info.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
            info.EnvironmentVariables["OPENBLAS_NUM_THREADS"] = "1";
            if (!useVenv)
                info.EnvironmentVariables["PYTHONPATH"] = dependencies;
            using (Process child = Process.Start(info))
            {
                if (child.WaitForExit(1000) && child.ExitCode != 0)
                    return Fail("The PureBlur window could not start with the local Python runtime.");
            }
            return 0;
        }
        catch (Exception error)
        {
            return Fail(error.Message);
        }
    }

    private static int Fail(string message)
    {
        MessageBox.Show(message, "PureBlur Studio", MessageBoxButtons.OK, MessageBoxIcon.Error);
        return 1;
    }
}
