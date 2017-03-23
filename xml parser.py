# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:53:09 2017

@author: Oakey
"""
fio=open("ID001_clinic_001.Temporal-Relation.gold.completed.xml","r")
fio=fio.read()
elem=[]
dec=1
i=0
test="Hello"

while dec != -1:
    dec=fio.find(';<span class="pl-ent">span</span>&gt;',dec)
    dec=dec+1
    dec=fio.find(';',dec)
    dec=dec+1
    dec2=fio.find(',',dec)
    stri=""
    while dec<dec2:
        stri=stri+fio[dec]
        dec=dec+1
    strii=int(stri)
    elem.append([])
    elem[i].append(strii)
    dec=dec+1
    dec2=fio.find('&',dec)
    stri=""
    while dec<dec2:
        stri=stri+fio[dec]
        dec=dec+1
    strii=int(stri)
    elem[i].append(strii)
    dec=fio.find(';<span class="pl-ent">type</span>&gt;',dec)
    dec=dec+1
    dec=fio.find(";",dec)
    dec=dec+1
    dec2=fio.find("&",dec)
    stri=""
    while dec<dec2:
        stri=stri+fio[dec]
        dec=dec+1
    elem[i].append(strii)
    dec2=fio.find(';<span class="pl-ent">span</span>&gt;',dec)
    dec=fio.find('<span class="pl-ent">Polarity</span>&gt;',dec)
    if dec<dec2:
        dec=fio.find(';',dec)
        dec=dec+1
        dec2=fio.find("&",dec)
        stri=""
        while dec<dec2:
            stri=stri=fio[dec]
            dec+1
        elem[i].append(stri)
    i=i+1
i=0
en=len(elem)
fi=file.open("",'r')
fi=fi.read()
lens=0
while i<en:   
    fi=fi[:elem[i][0]-2+lens]+"<"+elem[i][2]+">"+fi[:elem[i][0]+lens]
    lens=lens+len(elem[i][2])+2
    fi=fi[:elem[i][1]-2+lens]+"</"+elem[i][2]+">"+fi[:elem[i][1]+lens]
    lens=lens+len(elem[i][2])+3
    i=i+1
    
    