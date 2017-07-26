"""
Given a list of email addresses, go though the corpus, and 
get paths of each email if the email is sent to/from emails in the list.
Write the path to file name like:
from_someone@enron.com.txt
"""

import os
import re

FOLDER = '.'

def writeRecord(mailDir, writeFile):
    fn = os.path.join(FOLDER, writeFile)
    with open(fn,'a') as f:
        f.write(mailDir + '\n')
        
def analyseMail(mailAddr, mailHeader):
#given a mail header, return if mail's exist, in From or in To (include cc).
    from_s = mailHeader.find('From:')
    to_s = mailHeader.find('To:')
    from_string = mailHeader[from_s+6:to_s].strip()
    rest = mailHeader[to_s:]
    if mailAddr == from_string:
        return 'from'
    elif rest.find(mailAddr) > -1:
        return 'to'
    else:
        return None

def processFile(filename, emailAddrs):
    with open(filename,'r') as mail_f:
        content = mail_f.read()
        head = content.split('X-FileName')[0]
        for addr in emailAddrs:
            direction = analyseMail(addr,head)
            if direction != None:
                recordfile = "%s_%s.txt" % (direction,addr)
                writeRecord(filename,recordfile)
 
def walkAllMails(emailList):
    for dirpath, dirnames, filenames in os.walk('../maildir'): 
        for fn in filenames:
            fname = os.path.join(dirpath,fn)
            processFile(fname, emailList)
 
def example():
    mailList = ['cliff.baxter@enron.com']
    walkAllMails(mailList)
