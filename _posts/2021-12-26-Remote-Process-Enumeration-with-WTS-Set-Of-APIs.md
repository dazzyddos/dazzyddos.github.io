---
title: Remote Process Enumeration with WTS Set of Windows APIs
author: Dazzy Ddos
date: 2021-12-26 14:10:00 +0800
categories: [Enumeration, Red Teaming]
tags: [pentesting, hacking, red teaming, enumeration]

---

### Introduction

Hi All. I welcome you again. In this particular blog post we'll code our own tool in C++ to gather information (list of running processes) from remote system. We will be assuming that we got initial access in the AD network somehow and we want to gather information (in this case list of running processes) from remote system without having to use any complete framework tool with known signatures.

Windows API provides several ways to enumerate processes. The first set of APIs we will see are **ToolHelp** functions. They were introduced in Windows 2000 to faciliate easier process enumeration. ToolHelp comes with these set of APIs that can aid us in Process Enumeration, [**CreateToolhelp32Snapshot**](https://docs.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-createtoolhelp32snapshot), [**Process32First**](https://docs.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-process32first) and [**Process32Next**](https://docs.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-process32next). CreateToolhelp32Snapshot function allows us to enumerate both processes and threads. It also allows to enumeration modules and heaps in specific process. For process enumeration, Process32First and Process32Next function is going to be used. The first function will return the handle to the first process and we'll use Process32Next to enumeration through the list of process until we have no more processes.

Let's start writing the code to utilize the above API functions. **TlHelp32.h** is where all the ToolHelp32 functions defined.

```c
#include <Windows.h>
#include <stdio.h>
#include <TlHelp32.h>

int main()
{
	HANDLE hSnapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

	if (hSnapshot == INVALID_HANDLE_VALUE)
		return 1;

	PROCESSENTRY32 pe;
	pe.dwSize = sizeof(pe);

	if (::Process32First(hSnapshot, &pe)) {
		do {
			printf("PID: %u\tThreads: %u\tPPID: %u\tName: %ws\n", pe.th32ProcessID, pe.cntThreads, pe.th32ParentProcessID, pe.szExeFile);
		} while (::Process32Next(hSnapshot, &pe));
	}

	::CloseHandle(hSnapshot);

	return 0;
}
```

**CreateToolhelp32Snapshot** functions accepts two parameters, the first one is the flag which indicates what kind of enumeration we wish to do or what kind of snapshot we wish to capture, that could be either processes snapshot for the entire processes in the system or thread snapshot for the entire threads in the system or a set of modules or heaps in a particular process. We will stick with TH32CS_SNAPPROCESS to take the snapshot of processes. The second parameter takes the process id which is only relevant when we use the snapshot of heaps or modules of any specific process, we will keep this value 0 to acquire processes system wide. On success this function will return a valid handle to the snapshot. After that we are just if if in case we get returned an invalid handle, we will just return from the program with return value of 1.

**Process32First** function retrieves information about the first process encountered in a system snapshot. The first handle is the handle to the snapshot we had been returned from the **CreateToolhelp32Snapshot** and the second value is the structure where we get the result back called [**PROCESSENTRY32**](https://docs.microsoft.com/en-us/windows/win32/api/tlhelp32/ns-tlhelp32-processentry32) The only thing we need to do before calling the Process32First function is to declare the structure and initialize it's first size member to the size of the structure.

**Process32Next** function retrieves information about the next process recorded in a system snapshot. We will use this function to iterate through all the processes untili there's no more.

Inside the printf we are just printing the process id, number of threads it has, the parent process id and finally the name of the process.

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/wtsinfo/processlist.png)

This is how we could enumerate processes on the local machine using Windows API. How about doing the same on a remote machine. There's another way of enumerating processes, this is using WTS set of functions. WTS functions are set of Windows Terminal Services APIs that are intended for terminal services environment, but work equally well in a local environment. 

```c
#include <Windows.h>
#include <stdio.h>
#include <WtsApi32.h>

#pragma comment(lib, "Wtsapi32")

int main(int argc, char** argv)
{

	CHAR* host = argv[1];

	HANDLE hServer = ::WTSOpenServerA(host);
	if (!hServer) {
		printf("Could not open a handle to the server %s\n", host);

		return 1;
	}

	printf("Opened a handle to the server %s : 0x%p\n", host, hServer);

	WTS_PROCESS_INFOA* info = NULL;

	DWORD count;
	if (!::WTSEnumerateProcessesA(hServer, 0, 1, &info, &count)) {
		printf("Could not enumerate process on the host %s\n", host);
		return 1;
	}

	printf("Found %d processes\n", count);
	for (DWORD i = 0; i < count; i++) {
		printf("PID: %u\tSession: %u\tName: %s\n", info[i].ProcessId, info[i].SessionId, info[i].pProcessName);
	}

	::WTSFreeMemory(info);

	return 0;
}
```

[**WTSOpenServerA**](https://docs.microsoft.com/en-us/windows/win32/api/wtsapi32/nf-wtsapi32-wtsopenservera) is used to open a handle to the remote host on which we want to enumeate the list of running processes.

[**WTSEnumerate**](https://docs.microsoft.com/en-us/windows/win32/api/wtsapi32/nf-wtsapi32-wtsenumerateprocessesa) can be used to retrieve information about the active processes on either a local machine or remote host. The first argument to this function is the handle to the remote host. The second and third parameter will be 0 and 1 according to the official documentation. The fourth parameter is the array of structure where the information will be returned and last parameter is the count of the returned by the function.

Then we are using the for loop to iterate through each structure and then print the values one by one.

Let's run the program now to see if we can enumerate processes from the remote host (In my case I will be my DC's IP Address)

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/wtsinfo/Pasted%20image%2020211226111802.png))

You can see that we were successfully able to open the handle to the host but we could not enumerate running process which actually makes sense as our program is running in the context of normal domain user which doesn't have right to enumerate process on the Domain Controller.

We are going to make our program perform impersonation to get the access token of the user which has proper rights (In my case Domain Admin). I am going to copy paste the snippet of code from the [**scshell**](https://github.com/Mr-Un1k0d3r/SCShell/blob/master/SCShell.c) program


```c
CHAR* host = argv[1];
CHAR* domain = argv[2];
CHAR* username = argv[3];
CHAR* password = argv[4];
BOOL bResult = FALSE;

HANDLE hToken = NULL;

if (username != NULL) {
    printf("Username was provided attempting to call LogonUserA\n");

    bResult = LogonUserA(username, domain, password, LOGON32_LOGON_NEW_CREDENTIALS, LOGON32_PROVIDER_DEFAULT, &hToken);

if (!bResult) {

	 printf("LogonUserA failed %ld\n", GetLastError());
	
	 ExitProcess(0);

 }

bResult = FALSE;

bResult = ImpersonateLoggedOnUser(hToken);

if (!bResult) {

	 printf("ImpersonateLoggedOnUser failed %ld\n", GetLastError());
	
	 ExitProcess(0);

}
```

The above code takes Domain Name, Username and Password from the command line and tries to login and impersonate as that user. We'll add this code in our program to impersonate as the Domain Admin User to get process list from the Domain Controller.

So our final code will look like this now.

```c
#include <Windows.h>
#include <stdio.h>
#include <WtsApi32.h>

#pragma comment(lib, "Wtsapi32")

int main(int argc, char** argv)

{


	 CHAR* host = argv[1];
	 CHAR* domain = argv[2];
	 CHAR* username = argv[3];
	 CHAR* password = argv[4];
	 BOOL bResult = FALSE;
	
	 HANDLE hToken = NULL;
	
	 if (username != NULL) {
	
	 printf("Username was provided attempting to call LogonUserA\n");
	
	 bResult = LogonUserA(username, domain, password, LOGON32_LOGON_NEW_CREDENTIALS, LOGON32_PROVIDER_DEFAULT, &hToken);
	
	 if (!bResult) {
	
	 printf("LogonUserA failed %ld\n", GetLastError());
	
	 ExitProcess(0);
	
	 }
	
	 }
	
	 bResult = FALSE;
	
	 bResult = ImpersonateLoggedOnUser(hToken);
	
	 if (!bResult) {
	
	 printf("ImpersonateLoggedOnUser failed %ld\n", GetLastError());
	
	 ExitProcess(0);
	
	 }
	
	 HANDLE hServer = ::WTSOpenServerA(host);
	
	 if (!hServer) {
	
	 printf("Could not open a handle to the server %s\n", host);
	
	 return 1;
	
	 }
	
	 printf("Opened a handle to the server %s : 0x%p\n", host, hServer);
	
	 WTS_PROCESS_INFOA* info = NULL;
	
	
	 DWORD count;
	
	 if (!::WTSEnumerateProcessesA(hServer, 0, 1, &info, &count)) {
	
	 printf("Could not enumerate process on the host %s\n", host);
	
	 return 1;
	
	 }
	
	 printf("Found %d processes\n", count);
	
	 for (DWORD i = 0; i < count; i++) {
	
	 printf("PID: %u\tSession: %u\tName: %s\n", info[i].ProcessId, info[i].SessionId, info[i].pProcessName);
	
	 }
	
	 ::WTSFreeMemory(info);
	
	 return 0;

}
```

Now we can successfully enumerate process in Domain Controller.

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/wtsinfo/Pasted%20image%2020211226123550.png)
