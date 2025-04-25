---
title: TryHackMe Enterprise Walkthrough
author: Dazzy Ddos
date: 2021-03-22 14:10:00 +0800
categories: [Walkthrough]
tags: [pentesting, hacking, tryhackme, walkthrough]

---

[**Enterprise**](https://tryhackme.com/room/enterprise) is an awesome box from [TryHackMe](https://tryhackme.com/) by @NekoS3c

***
>You just landed in an internal network. You scan the network and there's only the Domain Controller...



## Enumeration

As always, we'll start with nmap scan. First all port scan and then version detection / default script nmap scan on the open ports.

```markdown
Nmap scan report for 10.10.157.225
Host is up, received user-set (0.16s latency).
Scanned at 2021-03-20 07:19:46 EDT for 547s
Not shown: 65507 closed ports
Reason: 65507 resets
PORT      STATE SERVICE          REASON
53/tcp    open  domain           syn-ack ttl 127
80/tcp    open  http             syn-ack ttl 127
88/tcp    open  kerberos-sec     syn-ack ttl 127
135/tcp   open  msrpc            syn-ack ttl 127
139/tcp   open  netbios-ssn      syn-ack ttl 127
389/tcp   open  ldap             syn-ack ttl 127
445/tcp   open  microsoft-ds     syn-ack ttl 127
464/tcp   open  kpasswd5         syn-ack ttl 127
593/tcp   open  http-rpc-epmap   syn-ack ttl 127
636/tcp   open  ldapssl          syn-ack ttl 127
3268/tcp  open  globalcatLDAP    syn-ack ttl 127
3269/tcp  open  globalcatLDAPssl syn-ack ttl 127
3389/tcp  open  ms-wbt-server    syn-ack ttl 127
5985/tcp  open  wsman            syn-ack ttl 127
7990/tcp  open  unknown          syn-ack ttl 127
9389/tcp  open  adws             syn-ack ttl 127
47001/tcp open  winrm            syn-ack ttl 127
49664/tcp open  unknown          syn-ack ttl 127
49665/tcp open  unknown          syn-ack ttl 127
49666/tcp open  unknown          syn-ack ttl 127
49667/tcp open  unknown          syn-ack ttl 127
49669/tcp open  unknown          syn-ack ttl 127
49670/tcp open  unknown          syn-ack ttl 127
49672/tcp open  unknown          syn-ack ttl 127
49675/tcp open  unknown          syn-ack ttl 127
49678/tcp open  unknown          syn-ack ttl 127
49684/tcp open  unknown          syn-ack ttl 127
49690/tcp open  unknown          syn-ack ttl 127

Read data files from: /usr/bin/../share/nmap
Nmap done: 1 IP address (1 host up) scanned in 547.23 seconds
```

```markdown
Host is up (0.16s latency).

PORT     STATE SERVICE       VERSION
53/tcp   open  domain?
| fingerprint-strings:
|   DNSVersionBindReqTCP:
|     version
|_    bind
80/tcp   open  http          Microsoft IIS httpd 10.0
| http-methods:
|_  Potentially risky methods: TRACE
|_http-server-header: Microsoft-IIS/10.0
|_http-title: Site doesn't have a title (text/html).
135/tcp  open  msrpc         Microsoft Windows RPC
139/tcp  open  netbios-ssn   Microsoft Windows netbios-ssn
445/tcp  open  microsoft-ds?
3389/tcp open  ms-wbt-server Microsoft Terminal Services
| rdp-ntlm-info:
|   Target_Name: LAB-ENTERPRISE
|   NetBIOS_Domain_Name: LAB-ENTERPRISE
|   NetBIOS_Computer_Name: LAB-DC
|   DNS_Domain_Name: LAB.ENTERPRISE.THM
|   DNS_Computer_Name: LAB-DC.LAB.ENTERPRISE.THM
|   DNS_Tree_Name: ENTERPRISE.THM
|   Product_Version: 10.0.17763
|_  System_Time: 2021-03-20T11:23:00+00:00
| ssl-cert: Subject: commonName=LAB-DC.LAB.ENTERPRISE.THM
| Not valid before: 2021-03-11T02:11:05
|_Not valid after:  2021-09-10T02:11:05
|_ssl-date: 2021-03-20T11:23:14+00:00; +1s from scanner time.
1 service unrecognized despite returning data. If you know the service/version, please submit the following fingerprint at https://nmap.org/cgi-bin/submit.cgi?new-service :
SF-Port53-TCP:V=7.80%I=7%D=3/20%Time=6055DA8A%P=x86_64-pc-linux-gnu%r(DNSV
SF:ersionBindReqTCP,20,"\0\x1e\0\x06\x81\x04\0\x01\0\0\0\0\0\0\x07version\
SF:x04bind\0\0\x10\0\x03");
Service Info: OS: Windows; CPE: cpe:/o:microsoft:windows

Host script results:
| smb2-security-mode:
|   2.02:
|_    Message signing enabled and required
| smb2-time:
|   date: 2021-03-20T11:23:03
|_  start_date: N/A

Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 164.22 seconds
```

### Web Enumeration

We found few *dns* names and DC hostname from the nmap output.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/01.png)

Let's add it to our */etc/hosts* file.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/02.png)

Nothing seems to be there on the website.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/03.png)

I also ran *gobuster* and *nikto* in the background.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/04.png)

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/05.png)

When visiting the website even with the respect dns names which we had found yields the same result.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/06.png)

### SMB Enumeration

We found some open shares.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/07.png)

Let's connect to the *Users* Share and download everything for what we have permission.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/08.png)

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/09.png)

Since i can see that there's *bitbucket* directory, so my next step was to see if there's any wordlist containing common files or directory names for the bitbucket inside **SecLists** 
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/10.png)

We couldn't find any wordlist for *bitbucket*. At this point of time, I was clueless and was deciding if i should do subdomain bruteforce or not. I went in again to check for all open ports and realized that i had missed one port i.e., **7990**. My nmap all port scan had found it but i forgot to include it in my nmap server version scan. So, again i ran nmap service version scan only for this port.
```bash
╰─➤  nmap -sC -sV -p7990 -T4 -Pn -n 10.10.86.5
Starting Nmap 7.80 ( https://nmap.org ) at 2021-03-22 09:19 EDT
Nmap scan report for 10.10.86.5
Host is up (0.18s latency).

PORT     STATE SERVICE VERSION
7990/tcp open  http    Microsoft IIS httpd 10.0
| http-methods:
|_  Potentially risky methods: TRACE
|_http-server-header: Microsoft-IIS/10.0
|_http-title: Log in to continue - Log in with Atlassian account
Service Info: OS: Windows; CPE: cpe:/o:microsoft:windows

Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 28.48 seconds
```

Opening it on browser takes it to 
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322092542.png)

The SSO links are empty and other links take us to the official atlassian website for login. At this point i don't have any valid usernames or email. Since it's a domain controller box and it has kerberos port open. So, i used **kerbrute** to find some valid usernames in the domain.
```markdwon
/opt/kerbrute/kerbrute_linux_amd64 userenum --dc 10.10.86.5 -d lab.enterprise.thm /opt/SecLists/Usernames/xato-net-10-million-usernames.txt -o users.txt
```

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322092800.png)

Now we have some valid usernames. I went ahead to use **Burp Intruder** to see if anyone of them is valid and taking us to the next step. Now at this exact point I realized, it's not making any *POST* request nor it's sending the data to any url.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322093401.png)

After submitting it's just loading the same page again. After this i went onto bruteforcing the subdomains but had no luck in that. 
The page is saying ``` Reminder to all Enterprise-THM Employees:  
We are moving to Github!``` . At this point i checked if they have any *.git* , *github*, *gitlab*, *bitbucket* or any such directories but they had not, then i went onto check if they have any **Github** repository.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322094307.png)

*@Sq00ky* is the creator for this machine. So at this point i knew that finally i am in the right direction.

Found one people in the repository.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322094814.png)

Found one powershell filed in his directory named **SystemInfo.ps1** which had two commits. When we revert back to the previous commit, we found his creds.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322094951.png)

Let's try if this creds are valid for *SMB* or *WinRM*
This user *nik* had only the same level of access as of the *anonymous* user.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322095414.png)

And he doesn't have access to the winrm either.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322095501.png)

Now since we have a valid domain creds, we can enumerate basic info using **rcpclient**
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322095950.png)

Let's add thesse users into our users.txt , many of which was already found by **kerbrute**.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322100313.png)

Now let's run **bloodhound.py** with the credentials we have.
I was getting the below error when running **bloodhound.py** although i am sure the arguments specified is correct. Maybe i will talk to the creator of this box on this and update the blog.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322101036.png)

Since i can't use bloodhound, my next step was to do everything manual. Starting with **ASPReproast** attack where we check if any user in the domain has pre-auth disabled using which we can request his *TGT* key which contains his password NTLM hash which we can try cracking locally.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322101339.png)

Next to see if any user has *SPN* set. If it is then we can request the *TGS* key since we are already part of the domain with the crednetials we have. The *TGS* key is encrypted with the password hash of the service. So, if we could crack it, we can get the password for the user.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322101748.png)

As can be seen, the user *bitbucket* has SPN set. So, now we can request TGS and try to crack it.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322101904.png)

Yayyyyy
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322102001.png)

Honestly I must say Withdrawing cash from ATM and cracking hashes successfully are the best two feelings, haha.

I tried the creds with smb and winrm but again no luck. At this point i again went back to the nmap port scan results and found the *RDP* port was also open. I first tried logging in using the *nik* user which failed and then tried the *bitbucket* user and this time success.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322102734.png)

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322102936.png)

We didn't have **Administrator** access. So my first step was privilege escalation. Did some basic enumeration(checking directories, files, current permission i.e., whoami /all etc) couldn't find anything so next step was to use **winPEAS.exe**.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322103428.png)

Not sure why there were no colors. Anyways, in the output we found that one service's binary path has no double quotes.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322103932.png)

It could be vulnerable to `Unquoted Service Path` if we have write access to any of those folders.

Bingo, We have write access :P
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322104555.png)

Now we'll use this script [Get-ServiceACL](https://rohnspowershellblog.wordpress.com/2013/03/19/viewing-service-acls/) to check if we have permission to start and stop the service.

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322105355.png)

As can be seen, we have permission to start and stop the service. Now, let's create our reverse shell using **msfvenom** and put it as *Zero.exe* inside that folder.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322105716.png)

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322105921.png)

Aaaaaaannnnddddd, we got our reverse shell back as *SYSTEM*, yayyyyyy.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/enterprise/Pasted%20image%2020210322110034.png)

It was a fun box again. Thanks for your time readers :)

