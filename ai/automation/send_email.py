#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
AUTOMATION
SEND EMAIL
Started on the 05/04/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import smtplib
import pandas as pd





config = pd.read_csv("C:/config.csv",sep = ";")
config.columns = ["key","value"]
config = config.set_index("key")

alert_settings = {
    "from_email":config.loc["gmail address"].iloc[0],
    "to_email":[config.loc["ekimetrics address"].iloc[0]],
    "password":config.loc["gmail password"].iloc[0]
}


gmail_settings = {
    "smtp":"smtp.gmail.com",
    "port":587
}



def send_email_alert(message,subject = "Alert mail - default"):
    send_email(message,subject,**alert_settings,**gmail_settings)


def send_email(message,subject = "DEFAUT MAIL SENDING", from_email = None, to_email = [], cc_mail = [], password = None,smtp='smtp.gmail.com', port=587):
    header  = 'From: %s\n' % from_email
    if len(to_email) > 0:
        header += 'To: %s\n' % ','.join(to_email)
    if len(cc_mail) > 0:
        header += 'Cc: %s\n' % ','.join(cc_mail)
    header += 'Subject: %s\n\n' % subject
    message = header + message

    server = smtplib.SMTP(smtp, port)  # use both smtpserver  and -port 
    server.starttls()
    server.login(from_email,password)
    problems = server.sendmail(from_email, to_email, message)
    server.quit()