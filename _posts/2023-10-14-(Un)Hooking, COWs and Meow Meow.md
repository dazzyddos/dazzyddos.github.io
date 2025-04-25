---
title: (Un)Hooking, COWs and Meow Meow
comments: true
author: Dazzy
date: 2023-10-14 08:15:00 +0800
categories:
  - Red Teaming
  - Defense Evasion
tags:
  - hooking
  - unhooking
  - frida
---

Konichiwa to all my readers! Today, I'm sharing a concise blog post centered on a query that arose during a malware development training session I was conducting. This post contains my observations and experiments in response to that question. Below, you'll find a screenshot of the exact query posed by one of the participants. At that time, I wasn't entirely sure of the answer, so I promised to research and circle back. To ensure I was on the right track and not diverging from the original question, I reached out to him today on Discord for a quick recap :P 
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014103148.png)

TLDR
>Yes, when you use NtMapViewOfFile and NtMapViewOfSection to load and map a copy of ntdll.dll from the disk into our process, and then attempt to modify its .text section of ntdll in memory with it's .text section, Copy On Write will indeed be triggered. This is because the memory pages of shared libraries like ntdll.dll are typically marked as read-only, and any write operation to these pages will invoke the COW. As a result of this, the specific page (or pages) we're trying to write to will be duplicated for our process, ensuring that other processes using the same ntdll.dll are not affected. 
>Now, regarding the question of whether two ntdll.dll instances would be loaded into your process, technically, yes. One instance is the original ntdll.dll that's loaded into every process by the OS. The second instance is the one you manually mapped using NtMapViewOfFile / NtMapViewOfSection. It's also important to understand that the manually mapped ntdll.dll will not be used by the system or other applications unless explicitly done so by your process [GetProcAddress(yourNtdll, "NtCreateFile")]


Now for those who wants to read through my struggles and noob debugger examples, please proceed to read through :P

When an OS loads shared libraries or system DLLs (like `ntdll.dll`), it tries to optimize memory usage. Instead of loading a new instance of the same library into memory for every process that requires it, the OS loads the library once and then maps it into the virtual address space of each process that uses it.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014103647.png)

This approach is memory-efficient, but there's a potential problem: what if one process tries to modify that shared library? If it was truly shared, this would modify the library for every process, which is undesirable. Let's test it out.
I am going to be using Frida to hook(patch) a specific function (NtCreateFile in our example) and let's see if it's been modified for all other process's as well
```javascript
var pNtCreateFile = Module.findExportByName("ntdll.dll", "NtCreateFile");

Interceptor.attach(pNtCreateFile, {
    onEnter: function (args) {
        send("[+] Called NtCreateFile [+]");
    }
});
```

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014104832.png)
From what we can see above, the `NtCreateFile` in the Notepad process got modified because of hooking, while the Calculator's stayed untouched, even though they're hanging out at the same memory address.

Copy on Write (CoW) is the answer. Initially, memory pages of the shared library are marked as read-only. When a process (like Notepad in our example) tries to write to a page of this shared memory, a page fault is generated because of the attempt to write to a read-only page. The OS handles this page fault by creating a private copy of modified page (not the whole DLL) for the writing process, then allowing the write to proceed on this private copy. Other processes still use the original shared page, and thus remain unaffected. This copied page replaces the shared page in your process's page table.

Let's look into the memory protection for the NTDLL and the NtCreateFile memory page in windbg.
Before hooking **NtCreateFile**
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014105438.png)

Checking protection of the memory page where the NtCreateFile is residing
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014105507.png)
Protection of the whole **ntdll** region
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014105549.png)

After hooking NtCreateFile
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014091126.png)

If we take a look at the protection of the ntdll region, you'll see that the whole ntdll region is not affected by COW. Only the section/page where the `NtCreateFile` resides is affected and modified.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014091301.png)

The memory in modern operating systems is managed in pages, typically of sizes like 4KB. When you modify a specific location in memory, only the page containing that location is affected. If the modification triggers a Copy-on-Write (CoW) event, only that specific page will be copied and made writable, not the entire region or module.
So, if you change bytes of the `NtCreateFile` API function in `ntdll.dll`, only the page (or pages) containing that API function will be subject to CoW. The rest of the `ntdll.dll` region will remain unchanged and continue to reference the original shared pages.
By only duplicating the modified pages rather than the entire module, the OS achieves efficiency in memory use, ensures that changes are isolated to the specific process making the modifications, and avoids unnecessary memory consumption, striking a balance between performance and the security of ensuring processes don't inadvertently affect one another.

Another question that you may ask now is how does our process(Notepad.exe) know which NtCreateFile to call, the one modified and private to our memory space or the one that resides in shared memory region?
The answer to this question was partially answered above in our statement 
>due to the Copy-on-Write (COW) mechanism, the OS makes a private copy of the modified page (not the whole DLL) just for your process. This copied page replaces the shared page in your process's PAGE TABLE.

Each process has its own virtual address space, and the OS, with the help of the MMU (Memory Management Unit), translates these virtual addresses to physical addresses. The page table is the core data structure used for this translation. When you modified `NtCreateFile`, the page table entry for that specific page was updated to point to the new, private page, while other entries remained unchanged and still point to shared pages. So, when your process calls the `NtCreateFile` function, it looks up the address in its own page table. Since the page table entry for that page now points to the private copy (thanks to your modification and COW), your process will execute the modified `NtCreateFile` function.

We can observe this behavior with `VMMap` (part of Sysinternals Suite) too
Before Hooking
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014093256.png)

After patching the `NtCreateFile` function, it's evident that the memory page where `NtCreateFile` resides has undergone a change. Its permissions have shifted from `PAGE_EXECUTE_READ` to `PAGE_EXECUTE_READWRITE`.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014093916.png)

So, I got a bit curious and wanted to try out some quirky experiments, even if they didn't seem to make a whole lot of sense. I created up a simple C code to load two `ntdll`s. Fun fact: if you try using `LoadLibrary` to load `ntdll` again, it won't bother if it spots the DLL already chilling in memory. So workaround would be to just copy `ntdll.dll` to a different folder and loaded it from there.
```c
#include <stdio.h>
#include <windows.h>

int main()
{
     printf("Before Loading another ntdll\n");
     getchar();
     HMODULE ntdllNew = LoadLibrary("C:\\Users\\hacke\\Desktop\\newntdll.dll");
     printf("After Loading another ntdll\n");
     printf("ntdllNew: %p\n", ntdllNew);
     printf("NtCreateFile of original NTDLL %p\n", GetProcAddress(GetModuleHandle("ntdll"),"NtCreateFile"));
     printf("NtCreateFile of new NTDLL %p\n", GetProcAddress(ntdllNew,"NtCreateFile"));
     getchar();

     return 0;
}
```

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014101345.png)

So, I went ahead and hooked `NtCreateFile` using Frida. Now, when I peeked inside the debugger, Only the `ntdll`'s `NtCreateFile` was modified. And it kinda makes sense cause both DLLs are hanging out in different regions of memory, and they aren't sharing the space. So, a change in one spot won't mess with the other.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014101507.png)

So now at this moment we know when you modify a page in memory, the operating system doesn't copy the entire `.text` region, but only the specific memory page that's being modified due to the Copy-On-Write (COW) mechanism. 
So if you're patching or modifying just one API function, like `NtCreateFile` or `MessageBoxA`, and it resides within a single page, only that specific page will be copied and made private to your process. If the modification spans multiple pages, then those specific pages will be copied. The rest of the `.text` region, and any other part of `ntdll.dll` that you haven't touched, will remain as shared read-only pages among all processes that use `ntdll.dll`.

Now to better understand the confirm the above theory, I took another ntdll unhooking example ired.team
```c
int main() {
	HANDLE process = GetCurrentProcess();
	MODULEINFO mi = {0};
	HMODULE ntdllModule = GetModuleHandleA("ntdll.dll");

     printf("Before UnHooking...\n");
     getchar();
	
	GetModuleInformation(process, ntdllModule, &mi, sizeof(mi));
	LPVOID ntdllBase = (LPVOID)mi.lpBaseOfDll;
	HANDLE ntdllFile = CreateFileA("C:\\Users\\hacke\\Desktop\\newntdll.dll", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
	HANDLE ntdllMapping = CreateFileMapping(ntdllFile, NULL, PAGE_READONLY | SEC_IMAGE, 0, 0, NULL);
	LPVOID ntdllMappingAddress = MapViewOfFile(ntdllMapping, FILE_MAP_READ, 0, 0, 0);

	PIMAGE_DOS_HEADER hookedDosHeader = (PIMAGE_DOS_HEADER)ntdllBase;
	PIMAGE_NT_HEADERS hookedNtHeader = (PIMAGE_NT_HEADERS)((DWORD_PTR)ntdllBase + hookedDosHeader->e_lfanew);

	for (WORD i = 0; i < hookedNtHeader->FileHeader.NumberOfSections; i++) {
		PIMAGE_SECTION_HEADER hookedSectionHeader = (PIMAGE_SECTION_HEADER)((DWORD_PTR)IMAGE_FIRST_SECTION(hookedNtHeader) + ((DWORD_PTR)IMAGE_SIZEOF_SECTION_HEADER * i));
		
		if (strcmp((char*)hookedSectionHeader->Name, ".text") == 0) {
			DWORD oldProtection = 0;
			BOOL isProtected = VirtualProtect((LPVOID)((DWORD_PTR)ntdllBase + (DWORD_PTR)hookedSectionHeader->VirtualAddress), hookedSectionHeader->Misc.VirtualSize, PAGE_EXECUTE_READWRITE, &oldProtection);
			memcpy((LPVOID)((DWORD_PTR)ntdllBase + (DWORD_PTR)hookedSectionHeader->VirtualAddress), (LPVOID)((DWORD_PTR)ntdllMappingAddress + (DWORD_PTR)hookedSectionHeader->VirtualAddress), hookedSectionHeader->Misc.VirtualSize);
			//isProtected = VirtualProtect((LPVOID)((DWORD_PTR)ntdllBase + (DWORD_PTR)hookedSectionHeader->VirtualAddress), hookedSectionHeader->Misc.VirtualSize, oldProtection, &oldProtection); // commented out to see the RWX region of .text
		}
	}

     printf("After Unhooking...\n");
     getchar();
	
	CloseHandle(process);
	CloseHandle(ntdllFile);
	CloseHandle(ntdllMapping);
	FreeLibrary(ntdllModule);
	
	return 0;
}
```
In the code above, we are going to be overwriting the complete .text region of the NTDLL section. Using VMMap to analyze the program, it became evident that no separate pages were generated this time. Instead, the entire `.text` region of `ntdll.dll` was duplicated locally for the process. The screenshot below shows the state of NTDLL .text region before overwriting (unhooking), it's evident that it's residing in the shared memory region.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231015113013.png)

After unhooking (overwriting) has been performed, the same .text region has now been copied locally and has now become a private region for our process.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231015113243.png)

The same behavior I observed with codes like Peruns Fart also where we modify the .text region of NTDLL in process.

I prepared few diagrams also who like to visualize things while learning like me :D

The diagram below is a representation of multiple processes reading from the same shared memory space of ntdll.dll library
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014230910.png)

The diagram below illustrates that when our process attempts a write operation (such as hooking or patching) on a specific API function address, the OS will generate a private local copy of the entire page containing that specific API function within our process's memory. Subsequently, the OS will adjust the page table to point to this locally mapped page. As a result, whenever our function invokes that specific API (in our instance, NtCreateFile), it will reference the local copy instead.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014231014.png)

"The diagram below illustrates how, when our process attempts to overwrite the .text region of a particular shared library, the OS, in response to a COW (Copy On Write) trigger, will create a dedicated local copy of the entire .text region within our process's memory.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231014231228.png)

#### Update (10/18/2023)

In response to Ali Hadi's insightful comment, I decided to delve deeper into the behavior of COW, specifically concerning the DLLs listed in the KnownDLLs object.

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231018103831.png)

**Background on KnownDLLs Object**
>KnownDLLs is a mechanism that Windows uses to optimize the loading of certain system libraries. DLLs listed in KnownDLLs are shared across all processes to speed up the system's performance. When a process needs to load a DLL, the system first checks if it's a KnownDLL. If it is, the system uses the already-loaded copy from the shared section instead of loading a new one from disk.

To address Ali's query, whether COW is exclusively triggered for DLLs present in KnownDLLs, I began by examining the content of the KnownDLLs object for my Windows

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231018103912.png)

Notably, `amsi.dll` was absent from the list. This was intriguing since `amsi.dll` is frequently utilized by many processes. Furthermore, I noticed the omission of `NTDLL` as well.
Taking `amsi.dll` as a test case, I reviewed its memory mapping using vmmap:
Before Hooking **AmsiScanBuffer**
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231018105104.png)

Subsequently, I hooked the `AmsiScanBuffer`. The memory mapping post-hooking revealed the creation of a new private memory map:
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/copyonwrite/Pasted%20image%2020231018105600.png)

**Conclusion:** The investigation above indicates that COW is not restricted to DLLs in the KnownDLLs object. Hooking a function in `amsi.dll` (which isn't part of KnownDLLs) led to COW behavior, evidenced by the creation of a new private memory map.
### Resources and References
https://security.stackexchange.com/questions/61771/is-api-hooking-done-by-a-process-in-a-shared-page-visible-to-all-other-processe

https://www.ired.team/offensive-security/defense-evasion/how-to-unhook-a-dll-using-c++
