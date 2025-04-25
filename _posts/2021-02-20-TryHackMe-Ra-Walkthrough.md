---
title: TryHackMe Ra Walkthrough
author: Dazzy Ddos
date: 2021-02-20 14:10:00 +0800
categories: [Walkthrough]
tags: [pentesting, hacking, tryhackme, walkthrough]

---

[**Ra**](https://tryhackme.com/room/ra) is an awesome box from [TryHackMe](https://tryhackme.com/) by @4nqr34z and @theart42.

## Port Scanning and Basic Enumeration

As always, will start with full port scan. Will do the other enumeration alongside till the nmap completes.

All open ports:

```python
Nmap scan report for 10.10.121.68
Host is up (0.17s latency).
Not shown: 65500 filtered ports
PORT      STATE SERVICE
53/tcp    open  domain
80/tcp    open  http
88/tcp    open  kerberos-sec
135/tcp   open  msrpc
139/tcp   open  netbios-ssn
389/tcp   open  ldap
445/tcp   open  microsoft-ds
464/tcp   open  kpasswd5
593/tcp   open  http-rpc-epmap
636/tcp   open  ldapssl
2179/tcp  open  vmrdp
3268/tcp  open  globalcatLDAP
3269/tcp  open  globalcatLDAPssl
3389/tcp  open  ms-wbt-server
5222/tcp  open  xmpp-client
5223/tcp  open  hpvirtgrp
5229/tcp  open  jaxflow
5262/tcp  open  unknown
5263/tcp  open  unknown
5269/tcp  open  xmpp-server
5270/tcp  open  xmp
5275/tcp  open  unknown
5276/tcp  open  unknown
5985/tcp  open  wsman
7070/tcp  open  realserver
7443/tcp  open  oracleas-https
7777/tcp  open  cbt
9090/tcp  open  zeus-admin
9091/tcp  open  xmltec-xmlmail
9389/tcp  open  adws
49670/tcp open  unknown
49672/tcp open  unknown
49673/tcp open  unknown
49674/tcp open  unknown
49694/tcp open  unknown
```

Ufff, that's a lot of ports. I did version scanning in the background. Meanwhile let's start our enumeration with port 80.

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/01.png)

There's nothing special on the website, all the tabs and links points to the same home page except for i found few emails.

![image](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/02.png)

Used curl to extract all those emails from the page.

![image3](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/03.png)

Stored them in a text file, hopefully it would be useful later.

In the source code, found a domain name, let's put it inside our hosts file.
![image4](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/04.png)

![image5](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/05.png)

## Reset Password of Lily

Now let's visit that **reset.asp** page mentioned in the source code.

![image6](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/06.png)

mmmm, looks like some reset password page, come'on dazzy, your readers are not fool.

So, let's play around with the page, maybe the emails we collected could come useful here. Meanwhile, i am also going to run gobuster in the background for directory bruteforcing, as **ippsec** says "there should always be something running in the background for enumeration"

```markdown
gobuster dir -u http://fire.windcorp.thm -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt -x asp
```

By the way, my nmap is still running :P

Anyways, let's continue. So, let me tell you what all i did in that reset page:
- First I tried all the usernames we had collected from the page and tried to bruteforce common names for the cars and common pet names using burpsuite intruder. It didn't work as planned.
- Tried SQL Injection but didn't work as well.

So after getting tired, i went to meet my girlfriend which obviously doesn't exist :pepeface: , wait why am i telling this to you :P

Anyways, so after that what i did was went little backwards and continued my web enumeration. I had completely forgotten that the webpage also consisted of employee names and images.
![employeeimage](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/07.png)

I first cross checked their names in the email list and it was not existing there, so i did the above steps again with these usernames now but it again didn't work out.

Let me tell you, i like puppies and pussies (obviously cat :P) . That puppy in the picture caugth my attention, i thought i could get some meta data from that image but it was easier, i got the name of the lady and her pet from the image name.
![ladywithpuppy](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/08.png)

So, we could finally reset the password with it.
![resetpass](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/09.png)

I have redacted the pass because i want you to follow with me you lazy **hacker** 

## SMB enumeration and First Flag

So, now we have the password, let's think what we could do with it, let's go way back to our enumeration phase and see what ports/services could be helpful here since i can't or maybe couldn't find any **CMS** login or admin or any type of login.

We have had **SMB** port open.

```markdown
445/tcp   open  microsoft-ds
```

Let's use **crackmapexec** to see if the pass we found is valid. We are using *cme* tool here because if the username lily doesn't work for the password we found, we can load the usernames from the email list we had previously grabbed.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/10.png)

As can be seen, that credentials was valid for the smb.
Now, let's see if we have any interesting files.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/11.png)

We got our first flag. There's some program residing in that directory. Google tells us it's some kind of Instant Messaging Software.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/12.png)

## Spark IM enumeration and exploitation

Let's go back to our port scan info and see if there's any matching service for it.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/13.png)

So, yeap, there's an Jabber service running at port 5222.
Let's install the exact version of **spark IM** on our machine.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/14.png)

Let's login with the creds we have.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/15.png)

I got some certificate error.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/16.png)

I then went to the advanced settings and disabled these options.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/17.png)

And then i was able to login. During the enumeration phase, i had come across the below website which mentions about the vulnerability in this specific version of spark IM.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/18.png)

```
Vulnerability Summary

An issue exists in Ignite Realtime Spark 2.8.3 (and the ROAR plugin for it) on Windows. A chat message can include an IMG element with a SRC attribute referencing an external host's IP address. Upon access to this external host, the (NT)LM hashes of the user are sent with the HTTP request. This allows an malicious user to collect these hashes, crack them, and potentially compromise the computer. (ROAR can be configured for automatic access. Also, access can occur if the user clicks.)
```

After googling for more, I came across an article by the official authors of this box. @4nqr34z and @theart42

```markdown
https://github.com/theart42/cves/blob/master/cve-2020-12772/CVE-2020-12772.md
```

It very well explains on how to leverage the vulnerability. Bottom line is we are sending an external url pointing to our machine which when clicked will send the user's NetNTLM hash to our **responder** listening for requests in the background.

Let's do it.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/19.png)

After waiting for a while, we got buse user's hash.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/20.png)

We cracked the hash with the john.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/21.png)

## User Shell and Second Flag

Wheeeeee, we have *winrm* access now and we got a user level shell.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/22.png)

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/23.png)

There were nothing inside those folders and files except for few images.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/24.png)

To be honest, i have full faith on my friend @4nqr34z and i am damn sure he won't make CTF style boxes. So, i didn't bother downloading and looking into those images, haha.

So, wandering through the file directories I found an interesting directory.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/25.png)
Hmmm, hmmm!!!

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/26.png)

hmmmmmmm
So, there seems to a script which runs every minute.
What caught my eyes from the script are the below lines in it.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/27.png)

In a nutshell, it reads from the **hosts.txt** file and each line runs through **Invoke-Express**

```
The `Invoke-Expression` cmdlet evaluates or runs a specified string as a command and returns the results of the expression or command. Without `Invoke-Expression`, a string submitted at the command line is returned (echoed) unchanged.

Source: Microsoft docs
```

So, now we need to write our commands somehow into **hosts.txt** file residing in **brittanycr**'s folder where we don't have permissions ofcourse *wink*. Those commands will get executed as **administrator**.

After little enumeration, I found that we belong to the **Account Operators** group.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/28.png)

```
The Account Operators group grants limited account creation privileges to a user. Members of this group can create and modify most types of accounts, including those of users, local groups, and global groups, and members can log in locally to domain controllers.

Source: Microsoft Docs
```

Let's now change the password for **brittanycr**
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/29.png)

I was successfull in changing the password but she didn't have winrm access to the box. So, i tried to login to smb using the creds we just updated.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/30.png)

And then went to her home directory.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/31.png)

I updated the contents of the **hosts.txt** file with the commands to create a new user **dazzy** and added him to **Administrators** group.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/32.png)

Let's overwrite the original file with this file.
![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/34.png)

<br>

## Admin shell and Third Flag

We were able to login as the admin user we created.

![](https://raw.githubusercontent.com/dazzyddos/dazzyddos.github.io/master/Images/ra/35.png)




