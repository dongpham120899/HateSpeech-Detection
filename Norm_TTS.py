import pandas as pd
import numpy as np
import re
from num2words import num2words
import string
from nltk import word_tokenize
from roman import fromRoman
import configparser
import csv
import pandas as pd
import copy
import nltk
nltk.download('punkt')

def remove_tag(str):
    cls = {'<PUNCT>': '</PUNCT>',
           '<MEASURE>': '</MEASURE>',
           '<CARDINAL>': '</CARDINAL>',
           '<DATE>': '</DATE>',
           '<TIME>': '</TIME>',
           '<VERBATIM>': '</VERBATIM>',
           '<ROMAN>': '</ROMAN>',
           '<DECIMAL>': '</DECIMAL>',
           '<ADDRESS>': '</ADDRESS>',
           '<FRACTION>': '</FRACTION>',
           '<ABBRE>': '</ABBRE>',
           '<FOREIGN>': '</FOREIGN>',
           '<DIGIT>': '</DIGIT>',
           '<LETTER>': '</LETTER>'}
    for tag_start, tag_end in cls.items():
        str = str.replace(tag_start, '')
        str = str.replace(tag_end, '')

    return str

# d="30-12-2001 là ngày 12/04/2003 23/10/95 13/02 1/2/96 30/04"
# print(re.findall(r"\b([0-9]{1,2})[-/:]([0-9]{1,2})[-/:]([0-9]{2,4})\b",d))
# print(re.findall(r"\b([0-9]{1,2})[-/:]([0-9]{1,2})\b",d))

def normalize_datetime(input_str, output_str):
    """
    Normalize datetime such as 12/12/2012 or 4/5/96'
    currently cannot differentiate sport score so only normalize 3 triplets 
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    type_1 = re.findall(r"([Nn]gày\s|[Hh]ôm\s|\b)([0-9]{1,2})[-/:]([0-9]{1,2})[-/:]([0-9]{2,4})+(\s|$)",input_str)
  
    if len(type_1) > 0:
        for item in type_1:

            ngay, day, month, year = item[0], item[1], item[2], item[3]
            
            string_0 = ngay+day+'/'+month+'/'+year
            string_1 = ngay+day+':'+month+':'+year
            string_2 = ngay+day+'-'+month+'-'+year
            
            if int(month) > 12:
                day, month = month, day
                
            day = ngay + day
            month = 'tháng ' + month
            year = 'năm ' + year
            
            replaced = day + ' ' + month + ' ' + year
            
            input_str = input_str.replace(string_0, replaced, 1)
            input_str = input_str.replace(string_1, replaced, 1)
            input_str = input_str.replace(string_2, replaced, 1)
            
            output_str = output_str.replace(string_0, replaced, 1)
            output_str = output_str.replace(string_1, replaced, 1)
            output_str = output_str.replace(string_2, replaced, 1) 
            
    type_2 = re.findall(r"([Nn]gày\s|[Hh]ôm\s|\b)([0-9]{1,2})[//]([0-9]{1,2})+(\s|$)", input_str)
    if len(type_2) > 0:
        for item in type_2:
            
            ngay, day, month = item[0], item[1], item[2]
            
            string_0 = ngay+day+'/'+month
            string_1 = ngay+day+':'+month
            string_2 = ngay+day+'-'+month
            
            if int(month) > 12:
                day, month = month, day
            
            if len(day) == 1:
                day = '0' + day
            if len(month) == 1:
                month = '0' + month
                
            day = ngay + day
            month = 'tháng ' + month
            
            replaced = day + ' ' + month
            
            input_str = input_str.replace(string_0, replaced, 1)
            input_str = input_str.replace(string_1, replaced, 1)
            input_str = input_str.replace(string_2, replaced, 1)
            
            output_str = output_str.replace(string_0, replaced, 1)
            output_str = output_str.replace(string_1, replaced, 1)
            output_str = output_str.replace(string_2, replaced, 1) 

    input_str = re.sub(r'\b(ngày)( \1\b)+', r'\1', input_str) #remove duplicated ngày in row
    output_str = re.sub(r'\b(ngày)( \1\b)+', r'\1', output_str) #remove duplicated ngày in row
    
    return input_str, output_str

def normalize_dottedwords(input_str, output_str):
    """
    Normalize sequences with forms a.b or a.b.c
    Currently under improvement
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    type_1 = re.findall('\s(\w+)\.(\w+)\.(?:(\w+))+(?=[\s]|$)', input_str)
    if len(type_1) > 0:
        for item in type_1:
            words = ''.join(item)
            replaced = '.'.join(item)
            input_str = input_str.replace(replaced, words)
            output_str = output_str.replace(replaced, words)
    
    type_2 = re.findall('\s(\w+)\.(\w+)+(?=[\s]|$)', input_str)
    if len(type_2) > 0:
        for item in type_2:
            words = ''.join(item)
            replaced = '.'.join(item)
            input_str = input_str.replace(replaced, words)
            output_str = output_str.replace(replaced, words)
            
    return input_str, output_str

def unit2words(input_str, output_str):
    # Units of information
    # input_str = input_str.replace('KB ', ' <MEASURE>KB</MEASURE> ')
    # input_str = input_str.replace('MB ', ' <MEASURE>MB</MEASURE> ')
    # input_str = input_str.replace('GB ', ' <MEASURE>GB</MEASURE> ')
    # input_str = input_str.replace('TB ', ' <MEASURE>TB</MEASURE> ')

    # 2G, 3G, etc
    input_str = input_str.replace(' 2G ', ' <MEASURE>2G</MEASURE> ')
    input_str = input_str.replace(' 3G ', ' <MEASURE>3G</MEASURE> ')
    input_str = input_str.replace(' 4G ', ' <MEASURE>4G</MEASURE> ')
    input_str = input_str.replace(' 5G ', ' <MEASURE>5G</MEASURE> ')

    # Units of frequency
    input_str = input_str.replace('GHz', ' <MEASURE>GHz</MEASURE> ')
    input_str = input_str.replace('MHz', ' <MEASURE>MHz</MEASURE> ')

    # Units of data-rate
    input_str = input_str.replace('Mbps', ' <MEASURE>Mbps</MEASURE> ')
    input_str = input_str.replace('Mb/s', ' <MEASURE>Mb/s</MEASURE> ')

    # Units of currency
    input_str = input_str.replace("đồng/", " <MEASURE>đồng/</MEASURE> ")
    input_str = input_str.replace("USD/", " <MEASURE>USD/</MEASURE> ")
    input_str = input_str.replace('đ ', ' <MEASURE>đ</MEASURE> ')
    input_str = input_str.replace('$', ' <MEASURE>$</MEASURE> ')
    input_str = input_str.replace('USD', ' <MEASURE>USD</MEASURE> ')
    input_str = input_str.replace('VNĐ', ' <MEASURE>VNĐ</MEASURE> ')
    input_str = input_str.replace('vnđ', ' <MEASURE>vnđ</MEASURE> ')
    input_str = input_str.replace('vnd', ' <MEASURE>vnd</MEASURE> ')
    input_str = input_str.replace('VND', ' <MEASURE>VND</MEASURE> ')

    # Units of area
    input_str = input_str.replace('km2', ' <MEASURE>km2</MEASURE> ')
    input_str = input_str.replace('cm2', ' <MEASURE>cm2</MEASURE> ')
    input_str = input_str.replace('mm2', ' <MEASURE>mm2</MEASURE> ')
    input_str = input_str.replace('m2', ' <MEASURE>m2</MEASURE> ')
    input_str = input_str.replace(' ha ', ' <MEASURE>ha</MEASURE> ')

    # Units of length
    # input_str = input_str.replace(' km ', ' <MEASURE>km</MEASURE> ')
    # input_str = input_str.replace(' cm ', ' <MEASURE>cm</MEASURE> ')
    # input_str = input_str.replace(' mm ', ' <MEASURE>mm</MEASURE> ')
    # input_str = input_str.replace(' nm ', ' <MEASURE>nm</MEASURE> ')
    input_str = input_str.replace('inch ', ' <MEASURE>inch</MEASURE> ')

    # Units of volume
    # input_str = input_str.replace('ml ', ' <MEASURE>ml</MEASURE> ')
    input_str = input_str.replace('cm3 ', ' <MEASURE>cm3</MEASURE> ')
    # input_str = input_str.replace('cc ', ' <MEASURE>cc</MEASURE> ')
    input_str = input_str.replace('m3 ', ' <MEASURE>m3</MEASURE> ')

    # Units of weight
    input_str = input_str.replace('/kg', ' <MEASURE>/kg</MEASURE> ')
    input_str = input_str.replace('kg/', ' <MEASURE>kg/</MEASURE> ')
    # input_str = input_str.replace('kg ', ' <MEASURE>kg</MEASURE> ')
    input_str = input_str.replace(' grams ', ' <MEASURE>grams</MEASURE> ')
    # input_str = input_str.replace(' mg ', ' <MEASURE>mg</MEASURE> ')

    # Units of temperature
    input_str = input_str.replace("oC ", " <MEASURE>oC</MEASURE> ")
    input_str = input_str.replace("ºC ", " <MEASURE>ºC</MEASURE> ")
    input_str = input_str.replace("ºF ", " <MEASURE>ºF</MEASURE> ")

    # Picture element
    # input_str = input_str.replace('MP ', ' <MEASURE>MP</MEASURE> ')

    # Units of speed
    input_str = input_str.replace("bpm", " <MEASURE>bpm</MEASURE> ")
    input_str = input_str.replace("nm/s", " <MEASURE>nm/s</MEASURE> ")
    input_str = input_str.replace("µm/s", " <MEASURE>µm/s</MEASURE> ")
    input_str = input_str.replace("mm/s", " <MEASURE>mm/s</MEASURE> ")
    input_str = input_str.replace("cm/s", " <MEASURE>cm/s</MEASURE> ")
    input_str = input_str.replace("dm/s", " <MEASURE>dm/s</MEASURE> ")
    input_str = input_str.replace("dam/s", " <MEASURE>dam/s</MEASURE> ")
    input_str = input_str.replace("hm/s", " <MEASURE>hm/s</MEASURE> ")
    input_str = input_str.replace("km/s", " <MEASURE>km/s</MEASURE> ")
    input_str = input_str.replace("m/s", " <MEASURE>m/s</MEASURE> ")
    input_str = input_str.replace("nm/giây", " <MEASURE>nm/giây</MEASURE> ")
    input_str = input_str.replace("µm/giây", " <MEASURE>µm/giây</MEASURE> ")
    input_str = input_str.replace("mm/giây", " <MEASURE>mm/giây</MEASURE> ")
    input_str = input_str.replace("cm/giây", " <MEASURE>cm/giây</MEASURE> ")
    input_str = input_str.replace("dm/giây", " <MEASURE>dm/giây</MEASURE> ")
    input_str = input_str.replace("dam/giây", " <MEASURE>dam/giây</MEASURE> ")
    input_str = input_str.replace("hm/giây", " <MEASURE>hm/giây</MEASURE> ")
    input_str = input_str.replace("km/giây", " <MEASURE>km/giây</MEASURE> ")
    input_str = input_str.replace("m/giây", " <MEASURE>m/giây</MEASURE> ")
    input_str = input_str.replace("nm/h", " <MEASURE>nm/h</MEASURE> ")
    input_str = input_str.replace("µm/h", " <MEASURE>µm/h</MEASURE> ")
    input_str = input_str.replace("mm/h", " <MEASURE>mm/h</MEASURE> ")
    input_str = input_str.replace("cm/h", " <MEASURE>cm/h</MEASURE> ")
    input_str = input_str.replace("dm/h", " <MEASURE>dm/h</MEASURE> ")
    input_str = input_str.replace("dam/h", " <MEASURE>dam/h</MEASURE> ")
    input_str = input_str.replace("hm/h", " <MEASURE>hm/h</MEASURE> ")
    input_str = input_str.replace("km/h", " <MEASURE>km/h</MEASURE> ")
    input_str = input_str.replace("kmh", " <MEASURE>kmh</MEASURE> ")
    input_str = input_str.replace("m/h", " <MEASURE>m/h</MEASURE> ")
    input_str = input_str.replace("nm/giờ", " <MEASURE>nm/giờ</MEASURE> ")
    input_str = input_str.replace("µm/giờ", " <MEASURE>µm/giờ</MEASURE> ")
    input_str = input_str.replace("mm/giờ", " <MEASURE>mm/giờ</MEASURE> ")
    input_str = input_str.replace("cm/giờ", " <MEASURE>cm/giờ</MEASURE> ")
    input_str = input_str.replace("dm/giờ", " <MEASURE>dm/giờ</MEASURE> ")
    input_str = input_str.replace("dam/giờ", " <MEASURE>dam/giờ</MEASURE> ")
    input_str = input_str.replace("hm/giờ", " <MEASURE>hm/giờ</MEASURE> ")
    input_str = input_str.replace("km/giờ", " <MEASURE>km/giờ</MEASURE> ")
    input_str = input_str.replace("m/giờ", " <MEASURE>m/giờ</MEASURE> ")

    # Others
    input_str = input_str.replace("/tấn", " <MEASURE>/tấn</MEASURE> ")
    input_str = input_str.replace("/thùng", " <MEASURE>/thùng</MEASURE> ")
    input_str = input_str.replace("/căn", " <MEASURE>/căn</MEASURE> ")
    input_str = input_str.replace("/cái", " <MEASURE>/cái</MEASURE> ")
    input_str = input_str.replace("/con", " <MEASURE>/con</MEASURE> ")
    input_str = input_str.replace("/năm", " <MEASURE>/năm</MEASURE> ")
    input_str = input_str.replace("/tháng", " <MEASURE>/tháng</MEASURE> ")
    input_str = input_str.replace("/ngày", " <MEASURE>/ngày</MEASURE> ")
    input_str = input_str.replace("/giờ", " <MEASURE>/giờ</MEASURE> ")
    input_str = input_str.replace("/phút", " <MEASURE>/phút</MEASURE> ")
    input_str = input_str.replace("đ/CP", " <MEASURE>đ/CP</MEASURE> ")
    input_str = input_str.replace("đ/lít", " <MEASURE>đ/lít</MEASURE> ")
    input_str = input_str.replace("đ/lượt", " <MEASURE>đ/lượt</MEASURE> ")
    input_str = input_str.replace("người/", " <MEASURE>người/</MEASURE> ")
    input_str = input_str.replace("giờ/", " <MEASURE>giờ/</MEASURE> ")
    input_str = input_str.replace('%', ' <MEASURE>%</MEASURE> ')
    input_str = input_str.replace('mAh ', ' <MEASURE>mAh</MEASURE> ')
    input_str = input_str.replace(" lít/", " <MEASURE>lít/</MEASURE> ")
    input_str = input_str.replace("./", "")
    input_str = input_str.replace("Nm", " <MEASURE>Nm</MEASURE> ")
    input_str = input_str.replace("º", " <MEASURE>º</MEASURE> ")
    input_str = input_str.replace("vòng 1/", " <MEASURE>vòng 1/</MEASURE> ")
    input_str = input_str.replace("mmol/l", " <MEASURE>mmol/l</MEASURE> ")
    input_str = input_str.replace("mg/", " <MEASURE>mg/</MEASURE> ")
    input_str = input_str.replace("triệu/", " <MEASURE>triệu/</MEASURE> ")
    input_str = input_str.replace("g/km", " <MEASURE>g/km</MEASURE> ")
    input_str = input_str.replace("ounce", " <MEASURE>ounce</MEASURE> ")
    input_str = input_str.replace("m3/s", " <MEASURE>m3/s</MEASURE> ")

    # Units of information
    # output_str = output_str.replace('KB ', ' <MEASURE>ki lô bai</MEASURE> ')
    output_str = output_str.replace('Mb ', ' <MEASURE>mê ga bai</MEASURE> ')
    output_str = output_str.replace('Gb ', ' <MEASURE>ghi</MEASURE> ')
    output_str = output_str.replace('GB ', ' <MEASURE>ghi ga bai</MEASURE> ')
    # output_str = output_str.replace('TB ', ' <MEASURE>tê ra bai</MEASURE> ')
    
    # 2G, 3G, etc
    output_str = output_str.replace(' 2G ', ' <MEASURE>hai gờ</MEASURE> ')
    output_str = output_str.replace(' 3G ', ' <MEASURE>ba gờ</MEASURE> ')
    output_str = output_str.replace(' 4G ', ' <MEASURE>bốn gờ</MEASURE> ')
    output_str = output_str.replace(' 5G ', ' <MEASURE>năm gờ</MEASURE> ')
    
    # Units of frequency
    output_str = output_str.replace('GHz', ' <MEASURE>ghi ga héc</MEASURE> ')
    output_str = output_str.replace('MHz', ' <MEASURE>mê ga héc</MEASURE> ')

    # Units of data-rate
    output_str = output_str.replace('Mbps', ' <MEASURE>mê ga bít trên giây</MEASURE> ')
    output_str = output_str.replace('Mb/s', ' <MEASURE>mê ga bít trên giây</MEASURE> ')
    
    # Units of currency
    output_str = output_str.replace("đồng/", " <MEASURE>đồng trên</MEASURE> ")
    output_str = output_str.replace("USD/", " <MEASURE>u ét đê trên</MEASURE> ")
#     output_str = output_str.replace('đ ', ' <MEASURE>đồng</MEASURE> ')
    output_str = output_str.replace('$', ' <MEASURE>đô la</MEASURE> ')
    output_str = output_str.replace('USD', ' <MEASURE>u ét đê</MEASURE> ')
    output_str = output_str.replace('VNĐ', ' <MEASURE>đồng</MEASURE> ')
    output_str = output_str.replace('vnđ', ' <MEASURE>đồng</MEASURE> ')
    output_str = output_str.replace('vnd', ' <MEASURE>đồng</MEASURE> ')
    output_str = output_str.replace('VND', ' <MEASURE>đồng</MEASURE> ')

    # Units of area
    output_str = output_str.replace('km2', ' <MEASURE>ki lô mét vuông</MEASURE> ')
    output_str = output_str.replace('cm2', ' <MEASURE>xen ti mét vuông</MEASURE> ')
    output_str = output_str.replace('mm2', ' <MEASURE>mi li mét vuông</MEASURE> ')
    output_str = output_str.replace('m2', ' <MEASURE>mét vuông</MEASURE> ')
#     output_str = output_str.replace(' ha ', ' <MEASURE>héc ta</MEASURE> ')
    
    # Units of length
    # output_str = output_str.replace(' km ', ' <MEASURE>ki lô mét</MEASURE> ')
    # output_str = output_str.replace(' cm ', ' <MEASURE>xen ti mét</MEASURE> ')
    # output_str = output_str.replace(' mm ', ' <MEASURE>mi li mét</MEASURE> ')
    # output_str = output_str.replace(' nm ', ' <MEASURE>na nô mét</MEASURE> ')
    output_str = output_str.replace('inch ', ' <MEASURE>inh</MEASURE> ')
    
    # Units of volume
    # output_str = output_str.replace('ml ', ' <MEASURE>mi li lít</MEASURE> ')
    output_str = output_str.replace('cm3 ', ' <MEASURE>xen ti mét khối</MEASURE> ')
    # output_str = output_str.replace('cc ', ' <MEASURE>xen ti mét khối</MEASURE> ')
    output_str = output_str.replace('m3 ', ' <MEASURE>mét khối</MEASURE> ')

    # Units of weight
    output_str = output_str.replace('/kg', ' <MEASURE>trên một ki lô gam</MEASURE> ')
    output_str = output_str.replace('kg/', ' <MEASURE>ki lô gam trên</MEASURE> ')
    # output_str = output_str.replace('kg ', ' <MEASURE>ki lô gam</MEASURE> ')
    output_str = output_str.replace(' grams ', ' <MEASURE>gờ ram</MEASURE> ')
    # output_str = output_str.replace(' mg ', ' <MEASURE>mi li gam</MEASURE> ')

    

    # Units of temperature
    output_str = output_str.replace("oC ", " <MEASURE>độ xê</MEASURE> ")
    output_str = output_str.replace("ºC ", " <MEASURE>độ xê</MEASURE> ")
    output_str = output_str.replace("ºF ", " <MEASURE>độ ép</MEASURE> ")
    
    # Picture element
    # output_str = output_str.replace('MP ', ' <MEASURE>mê ga píc xeo</MEASURE> ')
    
    # Units of speed
    output_str = output_str.replace("bpm", " <MEASURE>nhịp trên phút</MEASURE> ")
    output_str = output_str.replace("nm/s", " <MEASURE>na nô mét trên giây</MEASURE> ")
    output_str = output_str.replace("µm/s", " <MEASURE>mi cờ rô mét trên giây</MEASURE> ")
    output_str = output_str.replace("mm/s", " <MEASURE>mi li mét trên giây</MEASURE> ")
    output_str = output_str.replace("cm/s", " <MEASURE>xen ti mét trên giây</MEASURE> ")
    output_str = output_str.replace("dm/s", " <MEASURE>đề xi mét trên giây</MEASURE> ")
    output_str = output_str.replace("dam/s", " <MEASURE>đề ca mét trên giây</MEASURE> ")
    output_str = output_str.replace("hm/s", " <MEASURE>héc tô mét trên giây</MEASURE> ")
    output_str = output_str.replace("km/s", " <MEASURE>ki lô mét trên giây</MEASURE> ")
    output_str = output_str.replace("m/s", " <MEASURE>mét trên giây</MEASURE> ")
    output_str = output_str.replace("nm/giây", " <MEASURE>na nô mét trên giây</MEASURE> ")
    output_str = output_str.replace("µm/giây", " <MEASURE>mi cờ rô mét trên giây</MEASURE> ")
    output_str = output_str.replace("mm/giây", " <MEASURE>mi li mét trên giây</MEASURE> ")
    output_str = output_str.replace("cm/giây", " <MEASURE>xen ti mét trên giây</MEASURE> ")
    output_str = output_str.replace("dm/giây", " <MEASURE>đề xi mét trên giây</MEASURE> ")
    output_str = output_str.replace("dam/giây", " <MEASURE>đề ca mét trên giây</MEASURE> ")
    output_str = output_str.replace("hm/giây", " <MEASURE>héc tô mét trên giây</MEASURE> ")
    output_str = output_str.replace("km/giây", " <MEASURE>ki lô mét trên giây</MEASURE> ")
    output_str = output_str.replace("m/giây", " <MEASURE>mét trên giây</MEASURE> ")
    output_str = output_str.replace("nm/h", " <MEASURE>na nô mét trên giờ</MEASURE> ")
    output_str = output_str.replace("µm/h", " <MEASURE>mi cờ rô mét trên giờ</MEASURE> ")
    output_str = output_str.replace("mm/h", " <MEASURE>mi li mét trên giờ</MEASURE> ")
    output_str = output_str.replace("cm/h", " <MEASURE>xen ti mét trên giờ</MEASURE> ")
    output_str = output_str.replace("dm/h", " <MEASURE>đề xi mét trên giờ</MEASURE> ")
    output_str = output_str.replace("dam/h", " <MEASURE>đề ca mét trên giờ</MEASURE> ")
    output_str = output_str.replace("hm/h", " <MEASURE>héc tô mét trên giờ</MEASURE> ")
    output_str = output_str.replace("km/h", " <MEASURE>ki lô mét trên giờ</MEASURE> ")
    output_str = output_str.replace("kmh", " <MEASURE>ki lô mét trên giờ</MEASURE> ")
    output_str = output_str.replace("m/h", " <MEASURE>mét trên giờ</MEASURE> ")
    output_str = output_str.replace("nm/giờ", " <MEASURE>na nô mét trên giờ</MEASURE> ")
    output_str = output_str.replace("µm/giờ", " <MEASURE>mi cờ rô mét trên giờ</MEASURE> ")
    output_str = output_str.replace("mm/giờ", " <MEASURE>mi li mét trên giờ</MEASURE> ")
    output_str = output_str.replace("cm/giờ", " <MEASURE>xen ti mét trên giờ</MEASURE> ")
    output_str = output_str.replace("dm/giờ", " <MEASURE>đề xi mét trên giờ</MEASURE> ")
    output_str = output_str.replace("dam/giờ", " <MEASURE>đề ca mét trên giờ</MEASURE> ")
    output_str = output_str.replace("hm/giờ", " <MEASURE>héc tô mét trên giờ</MEASURE> ")
    output_str = output_str.replace("km/giờ", " <MEASURE>ki lô mét trên giờ</MEASURE> ")
    output_str = output_str.replace("m/giờ", " <MEASURE>mét trên giờ</MEASURE> ")

    # Others
    output_str = output_str.replace("/tấn", " <MEASURE>trên tấn</MEASURE> ")
    output_str = output_str.replace("/thùng", " <MEASURE>trên thùng</MEASURE> ")
    output_str = output_str.replace("/căn", " <MEASURE>trên căn</MEASURE> ")
    output_str = output_str.replace("/cái", " <MEASURE>trên cái</MEASURE> ")
    output_str = output_str.replace("/con", " <MEASURE>trên con</MEASURE> ")
    output_str = output_str.replace("/năm", " <MEASURE>trên năm</MEASURE> ")
    output_str = output_str.replace("/tháng", " <MEASURE>trên tháng</MEASURE> ")
    output_str = output_str.replace("/ngày", " <MEASURE>trên ngày</MEASURE> ")
    output_str = output_str.replace("/giờ", " <MEASURE>trên giờ</MEASURE> ")
    output_str = output_str.replace("/phút", " <MEASURE>trên phút</MEASURE> ")
    output_str = output_str.replace("đ/CP", " <MEASURE>đồng trên cổ phiếu</MEASURE> ")
    output_str = output_str.replace("đ/lít", " <MEASURE>đồng trên lít</MEASURE> ")
    output_str = output_str.replace("đ/lượt", " <MEASURE>đồng trên lượt</MEASURE> ")
    output_str = output_str.replace("người/", " <MEASURE>người trên</MEASURE> ")
    output_str = output_str.replace("giờ/", " <MEASURE>giờ trên</MEASURE> ")
    output_str = output_str.replace('%', ' <MEASURE>phần trăm</MEASURE> ')
    output_str = output_str.replace('mAh ', ' <MEASURE>mi li am pe</MEASURE> ')
    output_str = output_str.replace(" lít/", " <MEASURE>lít trên</MEASURE> ")
    output_str = output_str.replace("./", "")
    output_str = output_str.replace("Nm", " <MEASURE>Niu tơn mét</MEASURE> ")
    output_str = output_str.replace("º", " <MEASURE>độ</MEASURE> ")
    output_str = output_str.replace("vòng 1/", " <MEASURE>vòng 1</MEASURE> ")
    output_str = output_str.replace("mmol/l", " <MEASURE>mi li mon trên lít</MEASURE> ")
    output_str = output_str.replace("mg/", " <MEASURE>mi li gam trên</MEASURE> ")
    output_str = output_str.replace("triệu/", " <MEASURE>triệu trên</MEASURE> ")
    output_str = output_str.replace("g/km", " <MEASURE>gam trên ki lô mét</MEASURE> ")
    output_str = output_str.replace("ounce", " <MEASURE>ao</MEASURE> ")
    output_str = output_str.replace("m3/s", " <MEASURE>mét khối trên giây</MEASURE> ")
    
    return input_str, output_str

def money2words(input_str):
    number = input_str.split('k')[0]

    return num2words_fixed(number) +  ' ngàn'

def version2words(input_str):
    # Androi 2.2, 4.2.1...
    
    return input_str.replace('.', ' chấm ')
    
def multiply(input_str):

    return input_str.replace('x', ' nhân ')


def num2words_float(input_str):
    # Fix num2words for reading vietnamese float numbers
    l_part = input_str.split(',')[0]
    r_part = input_str.split(',')[1]
    l_part_str = num2words_fixed(l_part)
    
    if len(r_part) < 3:
        r_part_str = num2words_fixed(r_part)
        return l_part_str + ' phẩy ' + r_part_str
    else:
        r_part_str = ''
        for num in r_part:
            r_part_str += num2words(int(num), lang='vi') + ' '
        r_part_str.rstrip()

    return l_part_str + ' phẩy ' + r_part_str

def num2words_fixed(input_str):
    # Fix num2words for reading vietnamese numbers
    input_str = input_str.translate(str.maketrans('', '', string.punctuation))
    num2words_ = num2words(int(input_str), lang='vi')
    
    # Cases: 205-'hai tram le nam' --> 'hai tram linh nam'
    if 'trăm lẻ' in num2words_:
        num2words_ = num2words_.replace('trăm lẻ', 'trăm linh')
        # Cases: 2005-'hai nghin le nam' --> 'hai nghin khong tram linh nam'
    
    special_terms = ['lẻ một', 'lẻ hai', 'lẻ ba', 'lẻ bốn', 'lẻ năm', 'lẻ sáu', 'lẻ bảy','lẻ tám','lẻ chín']
    for term in special_terms:
        if num2words_.endswith(term):
            num2words_ = num2words_.replace('lẻ', 'không trăm linh')
            break
    
    # Cases: 2035-'hai nghin le ba muoi lam' --> 'hai nghin khong tram ba muoi lam'
    if 'lẻ' in num2words_:
        num2words_ = num2words_.replace('lẻ', 'không trăm')
    
    return num2words_


def date_dmy2words(input_str):
    day, month, year = re.findall(r'[\d]+', input_str)
    day_words = num2words_fixed(day)

    special_dates = ['một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy','tám','chín','mười']
    if day_words in special_dates:
        day_words = 'mùng ' + day_words

    month_words = num2words_fixed(month)
    
    if len(year) == 4:
        year_words = num2words(int(year), lang='vi')
    elif len(year) == 2:
        year_words = num2words(int('20'+year), lang='vi')
   
    year_words = num2words_fixed(year)
    output_str = day_words + ' tháng ' + month_words + ' năm ' + year_words
    
    return output_str

def date_dm2words(input_str):
    day, month = re.findall(r'[\d]+', input_str)
    day_words = num2words_fixed(day)
    month_words = num2words_fixed(month)
    special_dates = ['một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy','tám','chín','mười']
    if day_words in special_dates:
        day_words = 'mùng ' + day_words

    output_str = day_words + ' tháng ' + month_words
    
    return output_str

def date_my2words(input_str):
    month, year = re.findall(r'[\d]+', input_str)
    month_words = num2words_fixed(month)
    year_words = num2words_fixed(year)
    return month_words + ' năm ' + year_words

def phone2words(phone_number):
    phone_digits = re.findall(r'[0-9]', phone_number)
    for index, phone_digit in enumerate(phone_digits):
        phone_digit_str = num2words(int(phone_digit), lang='vi')
        phone_digits[index] = phone_digit_str

    phone_number_str = ' '.join(phone_digits)
    
    return phone_number_str


def time2words(time):
    if (time.find(':',3,6) != -1):
        time_hour = time.split(':')[0] 
        time_minute = time.split(':')[1]
        time_second = time.split(':')[2]
        time_hour_str = num2words(int(time_hour), lang='vi')
        time_minute_str = num2words(int(time_minute), lang='vi')
        time_second_str = num2words(int(time_second), lang='vi')
        time_str = time_hour_str + ' giờ ' + time_minute_str + ' phút ' + time_second_str + ' giây'
    elif (time.find('h') != -1):
        time_hour = time.split('h')[0] 
        time_minute = time.split('h')[1]
        time_hour_str = num2words(int(time_hour), lang='vi')
        if time_minute == '' or time_minute == '00':
            return time_hour_str + ' giờ '
        else:
            time_minute_str = num2words(int(time_minute), lang='vi')
            time_str = time_hour_str + ' giờ ' + time_minute_str + ' phút'     
    elif (time.find(':') != -1):
        time_hour = time.split(':')[0] 
        time_minute = time.split(':')[1]
        time_hour_str = num2words(int(time_hour), lang='vi')
        if time_minute == '' or time_minute == '00':
            return time_hour_str + ' giờ '
        else:
            time_minute_str = num2words(int(time_minute), lang='vi')
            time_str = time_hour_str + ' giờ ' + time_minute_str + ' phút'

    return time_str




def replace_multi_space(str):
    return re.sub(' +', ' ', str)


def replace_str(str, start_norm, end_norm, term, start_tag, end_tag):
    left = str[:start_norm]
    right = str[end_norm:]
    out_str = left + ' ' + start_tag + term.strip() + end_tag + ' ' + right
    return replace_multi_space(out_str)


def tokenize(input_str):
    """
    Normalize punctuations: separate words and puctuations.
    Note that @, %, etc   will be changed, i.e. 'abc@gmail.com' --> 'abc @ gmail.com'
    """
    tokens = word_tokenize(input_str)
    input_str = " ".join(tokens)

    return input_str


def norm_punct(input_str, output_str):
    
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    
    input_str = re.sub('([.,!?()])', r' \1 ', input_str)
    input_str = re.sub('\s{2,}', ' ', input_str)
               
    output_str = re.sub('([.,!?()])', r' \1 ', output_str)
    output_str = re.sub('\s{2,}', ' ', output_str)


    return input_str, output_str


def norm_verbatim(e):
    if e == '#':
        e = e.replace(e, 'thăng')
    elif e == '$':
        e = e.replace(e, 'đô la')
    elif e == '%':
        e = e.replace(e, 'phần trăm')
    elif e == '&':
        e = e.replace(e, 'và')
    elif e == '*':
        e = e.replace(e, 'nhân')
    elif e == '+':
        e = e.replace(e, 'cộng')
    elif e == '<':
        e = e.replace(e, 'nhỏ hơn')
    elif e == '=':
        e = e.replace(e, 'bằng')
    elif e == '>':
        e = e.replace(e, 'lớn hơn')
    elif e == '@':
        e = e.replace(e, 'a còng')
    elif e == '^':
        e = e.replace(e, 'mũ')
    elif e == '\\':
        e = e.replace(e, 'trên')

    return e


def norm_tag_verbatim(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    verbatim = '# $ % & * + < = > @ ^'.split()
    for e in verbatim:
        e = e.strip()
        input_str = input_str.replace(' ' + e + ' ', ' <VERBATIM>' + e + '</VERBATIM> ')
        if e == '#':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'thăng' + '</VERBATIM> ')
        elif e == '$':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'đô la' + '</VERBATIM> ')
        elif e == '%':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'phần trăm' + '</VERBATIM> ')
        elif e == '&':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'và' + '</VERBATIM> ')
        elif e == '*':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'nhân' + '</VERBATIM> ')
        elif e == '+':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'cộng' + '</VERBATIM> ')
        elif e == '<':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'nhỏ hơn' + '</VERBATIM> ')
        elif e == '=':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'bằng' + '</VERBATIM> ')
        elif e == '>':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'lớn hơn' + '</VERBATIM> ')
        elif e == '@':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'a còng' + '</VERBATIM> ')
        elif e == '^':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'mũ' + '</VERBATIM> ')
        elif e == '\\':
            output_str = output_str.replace(' ' + e + ' ', ' <VERBATIM>' + 'trên' + '</VERBATIM> ')

    return input_str, output_str


def normalize_letters(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    match_in = re.search('(?=(\s[BCDĐFGHJKLMNPQRSTVWXZ]{2,50}\s))', input_str)
    match_out = re.search('(?=(\s[BCDĐFGHJKLMNPQRSTVWXZ]{2,50}\s))', output_str)
    while match_in is not None and match_out is not None:
        start_match_in = match_in.start(1)
        end_match_in = match_in.end(1)
        term_in = input_str[start_match_in:end_match_in]
        input_str = replace_str(input_str, start_match_in, end_match_in, term_in, '<LETTER>', '</LETTER>')

        start_match_out = match_out.start(1)
        end_match_out = match_out.end(1)
        term_out = output_str[start_match_out :end_match_out]
        term_norm_out = term_out.replace(term_out , ' '.join(list(term_out )))
        output_str = replace_str(output_str, start_match_out , end_match_out , term_norm_out , '<LETTER>', '</LETTER>')

        match_in = re.search('(?=(\s[BCDĐFGHJKLMNPQRSTVWXZ]{2,50}\s))', input_str)
        match_out = re.search('(?=(\s[BCDĐFGHJKLMNPQRSTVWXZ]{2,50}\s))', output_str)

    return input_str, output_str


def normalize_AZ09(input_str, output_str):
    """
    Normalize sequences with forms [A-Z]{1}[0-9]{1,2} or [0-9]{1,2}[A-Z]{1},
    i.e. 'chung cư A10', 'cục phòng chống tội phạm công nghệ cao C50'
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    type_1 = re.findall('\s[A-Za-z]\d{1,2}\s', input_str)
    type_2 = re.findall('\s\d{1,2}[A-Za-z]\s', input_str)
    if len(type_1) > 0:
        for item in type_1:
            AZ = item[:2]
            num = item[2:]
            input_str = input_str.replace(item, AZ + ' ' + num)
            output_str = output_str.replace(item, AZ + ' ' + num)
    if len(type_2) > 0:
        for item in type_2:
            AZ = item[-2:]
            num = item[:-2]
            input_str = input_str.replace(item, num + ' ' + AZ)
            output_str = output_str.replace(item, num + ' ' + AZ)

    return input_str, output_str


def normalize_09unit(input_str, output_str):
    """
    Normalize sequences with forms [0-9]{n}[unit]{1},
    i.e. 'chung cư 10m', 'con em nay 3t'
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    type_1 = re.findall('\s\d{1,9}[A-Za-zđ]\s', input_str)
    if len(type_1) > 0:
        for item in type_1:
            AZ = item[-2:]
            num = item[:-2]
            if AZ == 't ': AZ = 'tuổi '
            elif AZ == 'đ ': AZ = 'đồng '
            elif AZ == 'm ': AZ = 'mét '
            elif AZ == 'p ': AZ = 'phút '
            elif AZ == 'k ': AZ = 'ka '
            input_str = input_str.replace(item, num + ' ' + AZ)
            output_str = output_str.replace(item, num + ' ' + AZ)
    
    type_2 = re.findall('\s\d{1,9}.\d{1,9}[A-Za-zđ]\s', input_str)
    if len(type_2) > 0:
        for item in type_2:
            AZ = item[-2:]
            num = item[:-2]
            if AZ == 't ': AZ = 'tuổi '
            elif AZ == 'đ ': AZ = 'đồng '
            elif AZ == 'm ': AZ = 'mét '
            elif AZ == 'p ': AZ = 'phút '
            elif AZ == 'k ': AZ = 'ka '
            input_str = input_str.replace(item, num + ' ' + AZ)
            output_str = output_str.replace(item, num + ' ' + AZ)
            
    return input_str, output_str

def normalize_decimal(input_str, output_str):
    """
    Normalize decimal numbers,
    i.e. '10.5', '3,8'
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    type_1 = re.findall('\d{1,9}\.\d{1,9}', input_str)
    type_2 = re.findall('\d{1,9}\,\d{1,9}', input_str)
    if len(type_1) > 0:
        for item in type_1:
            item_split = item.split('.')
            left = item_split[0]
            right = item_split[1]
            input_str = input_str.replace(item, left + ' chấm ' + right)
            output_str = output_str.replace(item, left + ' chấm ' + right)
    if len(type_2) > 0:
        for item in type_2:
            item_split = item.split(',')
            left = item_split[0]
            right = item_split[1]
            input_str = input_str.replace(item, left + ' phẩy ' + right)
            output_str = output_str.replace(item, left + ' phẩy ' + right)

    return input_str, output_str


def norm_measure(str, config_norm):

    str = ' ' + str + ' '
    norm_str = norm_measure_generic(str, config_norm['km_pattern'], 'km', ' ki lô mét ')
    norm_str = norm_measure_generic(norm_str, config_norm['m_pattern'], 'm', ' mét')
    norm_str = norm_measure_generic(norm_str, config_norm['cm_pattern'], 'cm', ' xen ti mét ')
    norm_str = norm_measure_generic(norm_str, config_norm['mm_pattern'], 'mm', ' mi li mét ')
    norm_str = norm_measure_generic(norm_str, config_norm['mm_pattern'], 'ms', ' mi li giây ')
    norm_str = norm_measure_generic(norm_str, config_norm['nm_pattern'], 'nm', ' na nô mét ')
    norm_str = norm_measure_generic(norm_str, config_norm['ha_pattern'], 'ha', ' héc ta ')
    norm_str = norm_measure_generic(norm_str, config_norm['l_pattern'], 'l', ' lít ')
    norm_str = norm_measure_generic(norm_str, config_norm['kg_pattern'], 'kg', ' ki lô gam ')
    norm_str = norm_measure_generic(norm_str, config_norm['g_pattern'], 'g', ' gam ')
    norm_str = norm_measure_generic(norm_str, config_norm['gr_pattern'], 'gr', ' gờ ram ')
    norm_str = norm_measure_generic(norm_str, config_norm['gram_pattern'], 'gram', ' gờ ram ')
    norm_str = norm_measure_generic(norm_str, config_norm['mg_pattern'], 'mg', ' mi li gam ')
    norm_str = norm_measure_generic(norm_str, config_norm['mmhg_pattern'], 'mmhg', ' mi li lít thủy ngân ')
    norm_str = norm_measure_generic(norm_str, config_norm['mmol_pattern'], 'mmol', ' mi li mon ')
    norm_str = norm_measure_generic(norm_str, config_norm['MP_pattern'], 'MP', ' mê ga píc xeo ')
    norm_str = norm_measure_generic(norm_str, config_norm['p_pattern'], 'p', ' phút ')
    norm_str = norm_measure_generic(norm_str, config_norm['s_pattern'], 's', ' giây ')
    norm_str = norm_measure_generic(norm_str, config_norm['m_odd_pattern'], 'm', ' mét ')
    norm_str = unit2words('', norm_str)[1]
    norm_str = remove_tag(norm_str)

    return norm_str

# print(norm_measure('   <MEASURE>ki lô mét trên giờ</MEASURE>    '))

def norm_measure_generic(str, pattern, term, repl):
    matches = re.findall(pattern, str)
    if len(matches) > 0:
        for item in matches:
            item_norm_out = item.replace(term, repl)
            str = str.replace(item, item_norm_out)

    return replace_multi_space(str)


def norm_tag_measure(input_str, output_str):
    """
    Normalize unit names and number + unit names. i.e.
    'kg' --> 'ki lô gam', '1000mAh' --> 'một nghìn mi li am pe'
    """
    # Normalize unit names (length, area, volume, information, speed, etc)
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '

    input_str, output_str = unit2words(input_str, output_str)

    km_pattern = '\s[0-9]*\.*\,*\-*[0-9]+km\s'
    m_pattern = '\s[0-9]*\.*\,*\-*[0-9]+\s*m\s'
    cm_pattern = '\s[0-9]*\.*\,*\-*[0-9]+cm\s'
    mm_pattern = '\s[0-9]*\.*\,*\-*[0-9]+mm\s'
    nm_pattern = '\s[0-9]*\.*\,*\-*[0-9]+nm\s'
    ha_pattern = '\s[0-9]*\.*\,*\-*[0-9]+ha\s'
    l_pattern = '\s[0-9]*\.*\,*\-*[0-9]+\s*L\s'
    kg_pattern = '\s[0-9]*\.*\,*\-*[0-9]+kg\s'
    g_pattern = '\s[0-9]*\.*\,*\-*[0-9]+\s*g\s'
    gr_pattern = '\s[0-9]*\.*\,*\-*[0-9]+gram\s'
    mg_pattern = '\s[0-9]*\.*\,*\-*[0-9]+mg\s'

    input_str, output_str = norm_tag_measure_generic(input_str, output_str, km_pattern, 'km ', ' ki lô mét ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, m_pattern, 'm ', ' mét ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, cm_pattern, 'cm ', ' xen ti mét ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, mm_pattern, 'mm ', ' mi li mét ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, nm_pattern, 'nm ', ' na nô mét ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, ha_pattern, 'ha ', ' héc ta ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, l_pattern, 'L ', ' lít ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, kg_pattern, 'kg ', ' ki lô gam ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, g_pattern, 'g ', ' gam ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, gr_pattern, 'gram ', ' gờ ram ')
    input_str, output_str = norm_tag_measure_generic(input_str, output_str, mg_pattern, 'mg ', ' mi li gam ')
    input_str, output_str = norm_soccer(input_str, output_str)

    return input_str, output_str


def norm_tag_measure_generic(input_str, output_str, pattern, term, norm_term):
    matches = re.findall(pattern, input_str)
    if len(matches) > 0:
        for item in matches:
            item_norm_out = item.replace(term, norm_term)
            input_str = input_str.replace(item, ' <MEASURE>' + item.strip() + '</MEASURE> ')
            output_str = output_str.replace(item, ' <MEASURE>' + item_norm_out + '</MEASURE> ')
    return input_str, output_str


def norm_soccer(input_str, output_str):
    # Normalize units of VFF football team: U23, U19, etc
    matches = re.findall('\sU[\-\.]*[0-9][0-9]\s', input_str)
    if len(matches) > 0:
        for item in matches:
            item_norm = item.replace('.','').replace('-','').replace(' U', ' U ')
            input_str = input_str.replace(item, '<MEASURE>' + item.strip() + '</MEASURE>')
            output_str = output_str.replace(item, item_norm)

    return input_str, output_str


def normalize_date(input_str, output_str):
    """
    Normalize dates.
    """
    input_str, output_str = norm_date_type_0(input_str, output_str)
    input_str, output_str = norm_date_type_1(input_str, output_str)
    input_str, output_str = norm_date_type_2(input_str, output_str)
    input_str, output_str = norm_date_type_3(input_str, output_str)
    input_str, output_str = norm_date_type_4(input_str, output_str)

    return input_str, output_str

def norm_date_type_0(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize dd/mm/yy[yy] (dmy) form of dates
    # Note: '8-6-2019' format này để riêng vì tránh cases "từ '8-6/2019' mây thay đổi nhiều"
    date_dmy_pattern = re.compile(r'([Nn]gày)+\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{2}|\d{4})\s|\s(0?[1-9]|[12]\d|3[01])[\-](0?[1-9]|[1][0-2])[\-](\d{2}|\d{4})\s')
    temp_str_date_dmy = input_str
    dates_dmy = []
    while(date_dmy_pattern.search(temp_str_date_dmy)):
        date = date_dmy_pattern.search(temp_str_date_dmy)
        term = date.group()
        x = re.search(r"\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{2}|\d{4})\s|\s(0?[1-9]|[12]\d|3[01])[\-](0?[1-9]|[1][0-2])[\-](\d{2}|\d{4})\s", term)
        dates_dmy.append(x.group())
        temp_str_date_dmy = temp_str_date_dmy[date.span()[1]-1:]

    if len(dates_dmy) > 0:
        for date in dates_dmy:
            date_str = date_dmy2words(date)
            # print('date_dd/mm/[yy]yy:', date, '-', input_str)
            input_str = input_str.replace(date, ' <DATE>' + date + '</DATE> ')
            output_str = output_str.replace(date, ' <DATE>' + date_str + '</DATE> ')

    return input_str, output_str

def is_date_type_1(str):
    str = ' ' + str + ' '
    pattern = re.compile(
        r'\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{2}|\d{4})\s|\s(0?[1-9]|[12]\d|3[01])[\-](0?[1-9]|[1][0-2])[\-](\d{2}|\d{4})\s')
    if pattern.search(str):
        return True
    return False


def norm_date_type_1(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize dd/mm/yy[yy] (dmy) form of dates
    # Note: '8-6-2019' format này để riêng vì tránh cases "từ '8-6/2019' mây thay đổi nhiều"
    date_dmy_pattern = re.compile(r'\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{2}|\d{4})\s|\s(0?[1-9]|[12]\d|3[01])[\-](0?[1-9]|[1][0-2])[\-](\d{2}|\d{4})\s')
    temp_str_date_dmy = input_str
    dates_dmy = []
    while(date_dmy_pattern.search(temp_str_date_dmy)):
        date = date_dmy_pattern.search(temp_str_date_dmy)
        dates_dmy.append(date.group())
        temp_str_date_dmy = temp_str_date_dmy[date.span()[1]-1:]

    if len(dates_dmy) > 0:
        for date in dates_dmy:
            date_str = date_dmy2words(date)
            # print('date_dd/mm/[yy]yy:', date, '-', input_str)
            input_str = input_str.replace(date, ' <DATE>' + date + '</DATE> ')
            output_str = output_str.replace(date, ' <DATE>' + 'ngày ' + date_str + '</DATE> ')

    return input_str, output_str


def is_date_type_2(str):
    str = ' ' + str + ' '
    pattern = re.compile(r'\s(0?[1-9]|[12]\d|3[01])[\/\-.](0?[1-9]|[1][0-2])\s')
    if pattern.search(str):
        return True
    return False


def norm_date_type_2(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize dd/mm (dm) form of dates
    date_dm_pattern = re.compile(r'(sau ,|mai ,|qua ,|nay ,|sớm|đến hết|[Đđ]ường|[Pp]hiên|[Nn]gày|[Ss]áng|[Tt]rưa|[Cc]hiều|[Tt]ối|[Đđ]êm|[Mm]ùng|[Hh]ôm|nay|[Ss]áng qua|[Tt]ưa qua|[Cc]hiều qua|[Tt]ối qua|[Đđ]êm qua|[Hh]ôm qua|[Hh]ôm sau|mai|[Vv]ào|kéo dài tới|dự kiến tới|đến|tới)\s\(*\s*(0?[1-9]|[12]\d|3[01])[\/\-.](0?[1-9]|[1][0-2])\s\)*')
    temp_str_date_dm = input_str
    dates_dm = []
    while(date_dm_pattern.search(temp_str_date_dm)):
        date = date_dm_pattern.search(temp_str_date_dm)
        if ')' in date.group():
            dates_dm.append(date.group().strip().split()[-2])
        else:
            dates_dm.append(date.group().strip().split()[-1])
        temp_str_date_dm = temp_str_date_dm[date.span()[1]-1:]

    # Cases: "số ra các ngày 28 29-2 và 1-3", "ngày 19 và 20.3 tới"
    dates_dm_special = re.findall('[Nn]gày .+ và (\d{1,2}[\/\-.]\d{1,2})\s', input_str)
    if len(dates_dm_special) > 0:
        dates_dm += dates_dm_special
    if len(dates_dm) > 0:
        for date in dates_dm:
            date_str = date_dm2words(date)
            # print('date_dd/mm:', date, '-', input_str)
            input_str = input_str.replace(' ' + date + ' ', ' <DATE>' + date + '</DATE> ')
            output_str = output_str.replace(' ' + date + ' ', ' <DATE>' + date_str + '</DATE> ')

    return input_str, output_str



def norm_date_type_3(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize dd/mm form without clear rules
    # Cám ơn cha dành cho Ngày của cha (16/6) tới đây của Hồ Ngọc Hà...
    p = re.compile(r'[Nn]gày .+\s\(*\s*(0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|[1][0-2])\s*\)*')
    l = []
    temp_line = input_str
    while(p.search(temp_line)):
        item = p.search(temp_line)
        l.append(item.group())
        temp_line = temp_line[item.span()[1]-1:]

    dates_dm_ = []
    if len(l) > 0:
        temp_str = l[0]
        p = re.compile(r'\s(0?[1-9]|[12]\d|3[01])\s*\/\s*(0?[1-9]|[1][0-2])\s')
        while(p.search(temp_str)):
            date = p.search(temp_str)
            dates_dm_.append(date.group())
            temp_str = temp_str[date.span()[1]-1:]

    if len(dates_dm_) > 0:
        for date in dates_dm_:
            date_str = date_dm2words(date)
            input_str = input_str.replace(date, ' <DATE>' + date + '</DATE> ')
            output_str = output_str.replace(date, ' <DATE>' + date_str + '</DATE> ')

    return input_str, output_str


def is_date_type_4(str):
    str = ' ' + str + ' '
    pattern = re.compile(r'\s(0?[1-9]|[1][0-2])[\/\-.](\d{4})\s')
    if pattern.search(str):
        return True
    return False


def norm_date_type_4(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize mm/yyyy (my) form of dates
    # @improve:
    # những cases không có [Tt]háng ở trước --> thêm 'tháng' ở date_my2word()
    # trong read.py  nhưng tránh các trường hợp Quý 2/2018, đợt 3/2019, tỷ lệ 1/2000
    date_my_pattern = re.compile(r'\s(0?[1-9]|[1][0-2])[\/\-.](\d{4})\s')
    temp_str_date_my = input_str
    dates_my = []
    while(date_my_pattern.search(temp_str_date_my)):
        date = date_my_pattern.search(temp_str_date_my)
        dates_my.append(date.group().strip())
        temp_str_date_my = temp_str_date_my[date.span()[1]-1:]
    if len(dates_my) > 0:
        for date in dates_my:
            date_str = date_my2words(date)
            # print('date_mm/yyyy:', date, '-', input_str)
            input_str = input_str.replace(' ' + date + ' ', ' <DATE>' + date + '</DATE> ')
            output_str = output_str.replace(' ' + date + ' ', ' <DATE>' + date_str + '</DATE> ')

    return input_str, output_str


def normalize_date_range(input_str, output_str):
    input_str, output_str = norm_date_range_type_1(input_str, output_str)
    input_str, output_str = norm_date_range_type_2(input_str, output_str)
    input_str, output_str = norm_date_range_type_3(input_str, output_str)
    input_str, output_str = norm_date_range_type_4(input_str, output_str)
    input_str, output_str = norm_date_range_type_5(input_str, output_str)
    input_str, output_str = norm_date_range_type_6(input_str, output_str)

    return input_str, output_str


def is_date_range_type_1(str):
    str = ' ' + str + ' '
    pattern = re.compile(
        r'\s(\d{4})\s*\-\s*(\d{4})\s')
    if pattern.search(str):
        return True
    return False


def norm_date_range_type_1(input_str, output_str):
    """
    Normalize date ranges.
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize yyyy-yyyy forms: 2016-2017, 1912-1982 ngày sinh, v.v.
    # @improve:
    # Khi nào thì chèn từ vào ví dụ 'năm học 2018-2019' thì đọc luôn tên năm còn
    # 'công ty thu thiếu hụt khoản này từ năm 2012-2017 là 11,3 tỷ đồng'
    # thì cần thêm từ "đến": từ năm 2012 đến 2017
    year_range_pattern = re.compile(r'\s(\d{4})\s*\-\s*(\d{4})\s')
    temp_str = input_str
    year_range_list = []
    while(year_range_pattern.search(temp_str)):
        year_range = year_range_pattern.search(temp_str)
        year_range_list.append(year_range.group())
        temp_str = temp_str[year_range.span()[1]-1:]

    if len(year_range_list) > 0:
        # print(str(year_range_list), " : ", input_str)
        for year_range in year_range_list:
            year_range_norm = year_range.replace('-', ' - ')
            year_range_norm = " ".join(year_range_norm.split())
            start_year = year_range_norm.split('-')[0]
            end_year = year_range_norm.split('-')[1]
            year_range_str = num2words_fixed(start_year) + ' đến ' + num2words_fixed(end_year)
            input_str = input_str.replace(year_range, ' <DATE>' + year_range + '</DATE> ')
            output_str = output_str.replace(year_range, ' <DATE>' + year_range_str + '</DATE> ')

    return input_str, output_str



# Normalize mm/yyyy-mm/yyyy forms

def is_date_range_type_2(str):
    str = ' ' + str + ' '
    pattern = re.compile(
        r'\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})\s')
    if pattern.search(str):
        return True
    return False


def norm_date_range_type_2(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize dd/mm/yyyy-dd/mm/yyyy forms
    date_range_dmy_pattern = re.compile(r'\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})\s')
    temp_str = input_str
    date_range_dmy_list = []
    while(date_range_dmy_pattern.search(temp_str)):
        date_range_dmy = date_range_dmy_pattern.search(temp_str)
        date_range_dmy_list.append(date_range_dmy.group())
        temp_str = temp_str[date_range_dmy.span()[1]-1:]

    if len(date_range_dmy_list) > 0:
        # print(str(date_range_dmy_list), " : ", input_str)
        for date_range_dmy in date_range_dmy_list:
            start_date = date_range_dmy.split('-')[0]
            end_date = date_range_dmy.split('-')[1]
            date_range_dmy_str = date_dmy2words(start_date) + ' đến ' + date_dmy2words(end_date)
            input_str = input_str.replace(date_range_dmy, ' <DATE>' + date_range_dmy + '</DATE> ')
            output_str = output_str.replace(date_range_dmy, ' <DATE>' + date_range_dmy_str + '</DATE> ')

    return input_str, output_str


def is_date_range_type_3(str):
    str = ' ' + str + ' '
    pattern = re.compile(
        r'\s(0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})\s')
    if pattern.search(str):
        return True
    return False


def norm_date_range_type_3(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize dd-dd/mm/yyyy forms
    date_range_dmy1_pattern = re.compile(r'\s(0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})\s')
    temp_str = input_str
    date_range_dmy1_list = []
    while(date_range_dmy1_pattern.search(temp_str)):
        date_range_dmy1 = date_range_dmy1_pattern.search(temp_str)
        date_range_dmy1_list.append(date_range_dmy1.group())
        temp_str = temp_str[date_range_dmy1.span()[1]-1:]

    if len(date_range_dmy1_list) > 0:
        # print(str(date_range_dmy1_list), " : ", input_str)
        for date_range_dmy1 in date_range_dmy1_list:
            start_date = date_range_dmy1.split('-')[0]
            end_date = date_range_dmy1.split('-')[1]
            date_range_dmy1_str = num2words_fixed(start_date) + ' đến ' + date_dmy2words(end_date)
            input_str = input_str.replace(date_range_dmy1, ' <DATE>' + date_range_dmy1 + '</DATE> ')
            output_str = output_str.replace(date_range_dmy1, ' <DATE>' + date_range_dmy1_str + '</DATE> ')

    return input_str, output_str


def is_date_range_type_4(str):
    str = ' ' + str + ' '
    pattern = re.compile(
        r'\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})\s')
    if pattern.search(str):
        return True
    return False


def norm_date_range_type_4(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize dd/mm-dd/mm/yyyy forms
    date_range_dmy2_pattern = re.compile(r'\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])[\/.](\d{4})\s')
    temp_str = input_str
    date_range_dmy2_list = []
    while(date_range_dmy2_pattern.search(temp_str)):
        date_range_dmy2 = date_range_dmy2_pattern.search(temp_str)
        date_range_dmy2_list.append(date_range_dmy2.group())
        temp_str = temp_str[date_range_dmy2.span()[1]-1:]

    if len(date_range_dmy2_list) > 0:
        # print(str(date_range_dmy2_list), " : ", input_str)
        for date_range_dmy2 in date_range_dmy2_list:
            start_date = date_range_dmy2.split('-')[0]
            end_date = date_range_dmy2.split('-')[1]
            date_range_dmy2_str = date_dm2words(start_date) + ' đến ' + date_dmy2words(end_date)
            input_str = input_str.replace(date_range_dmy2, ' <DATE>' + date_range_dmy2 + '</DATE> ')
            output_str = output_str.replace(date_range_dmy2, ' <DATE>' + date_range_dmy2_str + '</DATE> ')

    return input_str, output_str


def is_date_range_type_5(str):
    str = ' ' + str + ' '
    pattern = re.compile(
        r'\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s')
    if pattern.search(str):
        return True
    return False


def norm_date_range_type_5(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize dd/mm-dd/mm forms: 20/1-18/2
    date_range_dm1_pattern = re.compile(r'\s(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s')
    temp_str = input_str
    date_range_dm1_list = []
    while(date_range_dm1_pattern.search(temp_str)):
        date_range_dm1 = date_range_dm1_pattern.search(temp_str)
        date_range_dm1_list.append(date_range_dm1.group())
        temp_str = temp_str[date_range_dm1.span()[1]-1:]

    if len(date_range_dm1_list) > 0:
        # print(str(date_range_dm1_list), " : ", input_str)
        for date_range_dm1 in date_range_dm1_list:
            start_date = date_range_dm1.split('-')[0]
            end_date = date_range_dm1.split('-')[1]
            date_range_dm1_str = date_dm2words(start_date) + ' đến ' + date_dm2words(end_date)
            input_str = input_str.replace(date_range_dm1, ' <DATE>' + date_range_dm1 + '</DATE> ')
            output_str = output_str.replace(date_range_dm1, ' <DATE>' + date_range_dm1_str + '</DATE> ')

    return input_str, output_str


def is_date_range_type_6(str):
    str = ' ' + str + ' '
    pattern = re.compile(
        r'\s(0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s')
    if pattern.search(str):
        return True
    return False


def norm_date_range_type_6(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize dd-dd/mm forms: 15-18/6, 15 -18/6, 15- 18/6
    date_range_dm2_pattern = re.compile(r'\s(0?[1-9]|[12]\d|3[01])\s*\-\s*(0?[1-9]|[12]\d|3[01])[\/.](0?[1-9]|[1][0-2])\s')
    temp_str = input_str
    date_range_dm2_list = []
    while(date_range_dm2_pattern.search(temp_str)):
        date_range_dm2 = date_range_dm2_pattern.search(temp_str)
        date_range_dm2_list.append(date_range_dm2.group())
        temp_str = temp_str[date_range_dm2.span()[1]-1:]

    if len(date_range_dm2_list) > 0:
        # print(str(date_range_dm2_list), " : ", input_str)
        for date_range_dm2 in date_range_dm2_list:
            start_date = date_range_dm2.split('-')[0]
            end_date = date_range_dm2.split('-')[1]
            date_range_dm2_str = num2words(int(start_date), lang='vi') + ' đến ' + date_dm2words(end_date)
            input_str = input_str.replace(date_range_dm2, ' <DATE>' + date_range_dm2 + '</DATE> ')
            output_str = output_str.replace(date_range_dm2, ' <DATE>' + date_range_dm2_str + '</DATE> ')

    return input_str, output_str


def is_date_range(str):
    return is_date_range_type_1(str) or is_date_range_type_2(str) or is_date_range_type_3(str) \
           or is_date_range_type_4(str) or is_date_range_type_5(str) or is_date_range_type_6(str)

def norm_tt(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    p = re.compile(r"([\d]+\/[\d]+\/[A-Z\-*]+)|([\d]+\/[A-Z\-*]+)")
    temp_str = input_str
    number_plate_list = []
    while(p.search(temp_str)):
        number_plate = p.search(temp_str)
        x = number_plate.group().split("/")[:-1]
        n = "/".join(x)
        number_plate_list.append(n)
        temp_str = temp_str[number_plate.span()[1]-1:]
    if len(number_plate_list) > 0:
        for number_plate in number_plate_list:
            number_plate_str = phone2words(number_plate)
            # print('phone_number:', phone_number, '-', phone_number_str)
            input_str = input_str.replace(number_plate, ' <DIGIT>' + number_plate + '</DIGIT> ')
            output_str = output_str.replace(number_plate, ' <DIGIT>' + number_plate_str + '</DIGIT> ')

    return input_str, output_str

def norm_url(input_str, output_str):

    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    p = re.compile(r"[a-zA-Z0-9\.?_]+\s@\s+([a-zA-Z0-9\.?\-])+|\s[a-zA-Z0-9]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\-])*|((http|https)\s\:\s\/\/)+([a-zA-Z0-9\.?\-])+")  
    temp_str = input_str
    urls_list = []
    while(p.search(temp_str)):
        url = p.search(temp_str)
        term = url.group()
        urls_list.append(term)
        temp_str = temp_str[url.span()[1]-1:]
    # digits = re.findall(r"[\d ]{9,20}", input_str)
    if len(urls_list) > 0:
        for url in urls_list:
            extrac = version2words(url)
            input_str = input_str.replace(url, ' ' + url + ' ')
            output_str = output_str.replace(url, ' ' + extrac + ' ')

    return input_str, output_str

def norm_roman(roman_term):
    roman2int = fromRoman(roman_term)
    roman_numeral_str = num2words_fixed(roman2int)
    return roman_numeral_str


def norm_tag_roman_num(input_str, output_str):
    """
    Normalize roman numerals.
    """
    # @improve: 'Nữ hoàng Anh Elizabeth II thường đi lại trên chiếc xe của Land
    # Rovers và Jaguars' --> II đọc thành là 'đệ nhị'
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    roman_numeral_p = re.compile('\s(X{0,3})(IX|IV|V?I{0,3})\s|\s(x{0,3})(ix|iv|v?i{0,3})\s')

    # For double check if the string is roman numeral or not
    roman_numeral_check = re.compile('(thế hệ|số|đại hội|thứ|giai đoạn|quý|cấp|quận|kỳ|khóa|quy định|và|vành đai|vùng|thế kỷ|loại|khu vực|khu|đợt|hạng|báo động|tập|lần|trung ương|tw|chương)(\s(X{0,3})(IX|IV|V?I{0,3})\s|\s(x{0,3})(ix|iv|v?i{0,3})\s)', re.IGNORECASE)

    temp_str = input_str
    temp_str = " " + " ".join(word_tokenize(temp_str)) + " "
    roman_numeral_list = []
    while(roman_numeral_p.search(temp_str, re.IGNORECASE)):
        roman_numeral = roman_numeral_p.search(temp_str)
        # print("roman_numeral: ", roman_numeral)
        roman_numeral_list.append(roman_numeral.group().strip())
        temp_str = temp_str[roman_numeral.span()[1]-1:]

    if len(roman_numeral_list) > 0:
        # print('roman_numeral_list:', str(roman_numeral_list), " : ", input_str)
        for roman_numeral in roman_numeral_list:
            # if roman_numeral in ['X', 'V', 'x', 'v']:
            if (roman_numeral_check.search(input_str, re.IGNORECASE)):
                roman2int = fromRoman(roman_numeral.upper())
                roman_numeral_str = num2words(roman2int, lang='vi')
                input_str = input_str.replace(' ' + roman_numeral + ' ', ' <ROMAN>' + roman_numeral + '<ROMAN> ')
                output_str = output_str.replace(' ' + roman_numeral + ' ', ' <ROMAN>' + roman_numeral_str + '<ROMAN> ')
            # else:
            #     roman2int = fromRoman(roman_numeral.upper())
            #     roman_numeral_str = num2words(roman2int, lang='vi')
            #     input_str = input_str.replace(' ' + roman_numeral + ' ', ' <ROMAN>' + roman_numeral + '</ROMAN> ')
            #     output_str = output_str.replace(' ' + roman_numeral + ' ', ' <ROMAN>' + roman_numeral_str + '</ROMAN> ')

    return input_str, output_str


def norm_tag_roman_num_v2(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    roman_numeral_p = re.compile('\s(\(\s*X{0,3})(IX|IV|V?I{0,3})\s*\)\s|\s(\(\s*x{0,3})(ix|iv|v?i{0,3})\s*\)\s')
    temp_str = input_str
    temp_str = " " + " ".join(word_tokenize(temp_str)) + " "
    roman_numeral_list = []
    while (roman_numeral_p.search(temp_str, re.IGNORECASE)):
        roman_numeral = roman_numeral_p.search(temp_str)
        # print("roman_numeral: ", roman_numeral)
        roman_numeral_list.append(roman_numeral.group().strip())
        temp_str = temp_str[roman_numeral.span()[1] - 1:]
    if len(roman_numeral_list) > 0:
        for roman_numeral in roman_numeral_list:
            roman = roman_numeral.replace('(','').replace(')','').strip()
            roman2int = fromRoman(roman.upper())
            roman_numeral_str = num2words(roman2int, lang='vi')
            input_str = input_str.replace(' ' + roman_numeral + ' ', ' <ROMAN>' + roman_numeral + '<ROMAN> ')
            output_str = output_str.replace(' ' + roman_numeral + ' ', ' <ROMAN>, ' + roman_numeral_str + ' ,<ROMAN> ')

    return input_str, output_str


def normalize_phone_number(input_str, output_str):
    """
    Normalize phone numbers
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    p = re.compile(r"(hotline|tổng đài|điện thoại|đường dây nóng|liên hệ|gọi|call|chi tiết|hỗ trợ|tư vấn|công ty|hoặc)+\s*\:*\s*(\d{8,12}|\d{3}\s\d{4}\s\d{4}|\d{4}\s\d{3}\s\d{3,4}|\d{3,4}\s\d{6,7}|\d{3,4}\b)")
    temp_str = input_str
    phone_number_list = []
    while(p.search(temp_str)):
        phone_number = p.search(temp_str)
        term = phone_number.group()
        x = re.search(r"\d{8,12}|\d{3}\s\d{4}\s\d{4}|\d{4}\s\d{3}\s\d{3,4}|\d{3,4}\s\d{6,7}|\d{3,4}\b",term)
        phone_number_list.append(x.group())
        temp_str = temp_str[phone_number.span()[1]-1:]
    if len(phone_number_list) > 0:
        for phone_number in phone_number_list:
            phone_number_str = phone2words(phone_number)
            # print('phone_number:', phone_number, '-', phone_number_str)
            input_str = input_str.replace(' ' + phone_number + ' ', ' <DIGIT>' + phone_number + '</DIGIT> ')
            output_str = output_str.replace(' ' + phone_number + ' ', ' <DIGIT>' + phone_number_str + '</DIGIT> ')

    return input_str, output_str

def phone_single(input_str, output_str):
    """
        Normalize CMT, STK
        """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    p = re.compile(r"\d{8,12}|\d{2,5}\s\d{2,4}\s\d{2,4}\s\d{2,4}|\d{5}\.\d{5}|\d{3,5}\s\d{2,4}\s\d{2,4}|\d{3}\s\d{3}\s\d{4}|\d{3,5}\s\d{5,7}|\d{3}\.\d{4}\.\d{3}|\d{3}\.\d{3}\.\d{4}|\d{4}\.\d{3}\.\d{3}|\d{2}\.\d{4}\.\d{4}")
    # phone_numbers = re.findall(r"\d{10,12}|\d{4}\s\d{3}\s\d{3,4}|\d{3,4}\s\d{6,7}", input_str)
    # phone_numbers = re.findall(r"((09|03|07|08|05)+([0-9]{8})\b)", input_str)

    temp_str = input_str
    phone_number_list = []
    while(p.search(temp_str)):
        phone_number = p.search(temp_str)
        phone_number_list.append(phone_number.group())
        temp_str = temp_str[phone_number.span()[1]-1:]
    if len(phone_number_list) > 0:
        for phone_number in phone_number_list:
            phone_number_str = phone2words(phone_number)
            # print('phone_number:', phone_number, '-', phone_number_str)
            input_str = input_str.replace(phone_number, ' ' + ' <DIGIT>' + phone_number + '</DIGIT> ' + ' ')
            output_str = output_str.replace(phone_number, ' ' + ' <DIGIT>' + phone_number_str + '</DIGIT> ' + ' ')

    return input_str, output_str

def normalize_number_plate(input_str, output_str):
    """
        Normalize CMT, STK
        """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    p = re.compile(r"(([0-9]+[a-zA-Z]+[0-9])+\s*\-\s*[0-9]+\s)")
    temp_str = input_str
    number_plate_list = []
    while(p.search(temp_str)):
        number_plate = p.search(temp_str)
        x = number_plate.group()
        number_plate_list.append(x.split("-")[-1])
        temp_str = temp_str[number_plate.span()[1]-1:]
    if len(number_plate_list) > 0:
        for number_plate in number_plate_list:
            number_plate_str = phone2words(number_plate)
            # print('phone_number:', phone_number, '-', phone_number_str)
            input_str = input_str.replace(number_plate, ' <DIGIT>' + number_plate + '</DIGIT> ')
            output_str = output_str.replace(number_plate, ' <DIGIT>' + number_plate_str + '</DIGIT> ')

    return input_str, output_str

def norm_digit(input_str, output_str):
    """
        Normalize CMT, STK
        """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # p = re.compile(r"[\d ]{9,20}")
    p = re.compile(r"(chứng minh nhân dân|chứng minh thư|mã thẻ|số thẻ|số tài khoản|căn cước|mã số thuế|mã số|nhân viên|mã)+\s*\:*\s*(\d{2,20}\b)")  
    temp_str = input_str
    digits_list = []
    while(p.search(temp_str)):
        digit = p.search(temp_str)
        term = digit.group()
        x = re.search(r"\d{2,20}\b",term)
        digits_list.append(x.group())
        temp_str = temp_str[digit.span()[1]-1:]
    # digits = re.findall(r"[\d ]{9,20}", input_str)
    if len(digits_list) > 0:
        for digit in digits_list:
            print(digit)
            digits_str = phone2words(digit)
            # print('phone_number:', phone_number, '-', phone_number_str)
            input_str = input_str.replace(digit, ' <DIGIT>' + digit + '</DIGIT> ')
            output_str = output_str.replace(digit, ' <DIGIT>' + digits_str + '</DIGIT> ')

    return input_str, output_str

def normalize_negative_number(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # p = re.compile(r'\s\-([\d]+)\s')
    p = re.compile(r"(là|kết quả|âm|dưới|lạnh|xuống|nhiệt độ|áp suất)+\s*\:*\-\s*[0-9]*,*[0-9]+\s")
    temp_str = input_str
    neg_numbers = []
    while (p.search(temp_str)):
        numbers = p.search(temp_str)
        term = numbers.group()
        neg_numbers.append(term.split("-")[-1])
        temp_str = temp_str[numbers.span()[1] - 1:]
    if len(neg_numbers) > 0:
        for number in neg_numbers:
            if ',' in number:
                numbers_str = num2words_float(number)
            else:
                numbers_str = num2words_fixed(number)
            numbers_str = ' âm ' + numbers_str
            input_str = input_str.replace(number, ' <CARDINAL>' + number + '</CARDINAL> ')
            output_str = output_str.replace(number, ' <CARDINAL>' + numbers_str + '</CARDINAL> ')

    return input_str, output_str


def normalize_number_range(input_str, output_str):
    """
    Normalize number ranges
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    p = re.compile(r"(hơn|kém|gấp|tăng|tầm|giảm|nhất|tới|có|sau|mức|tuổi|từ|tăng tốc|được|khoảng)\s+[0-9]*,*[0-9]+\s*\-\s*[0-9]*,*[0-9]+\s")
    temp_str = input_str
    number_range_list = []
    while(p.search(temp_str)):
        number_range = p.search(temp_str)
        term = number_range.group()
        x = re.search(r"[0-9]*,*[0-9]+\s*\-\s*[0-9]*,*[0-9]+\s", term)
        number_range_list.append(x.group())
        temp_str = temp_str[number_range.span()[1]-1:]

    if len(number_range_list) > 0:
        # print(str(number_range_list), ' : ', input_str)
        for number_range in number_range_list:
            start_num = number_range.split('-')[0]
            end_num = number_range.split('-')[1]
            if ',' in start_num:
                start_num = num2words_float(start_num)
            else:
                start_num = num2words_fixed(start_num)

            if ',' in end_num:
                end_num = num2words_float(end_num)
            else:
                end_num = num2words_fixed(end_num)

            number_range_str = start_num + ' đến ' + end_num
            input_str = input_str.replace(number_range, ' <CARDINAL>' + number_range + '</CARDINAL> ')
            output_str = output_str.replace(number_range, ' <CARDINAL>' + number_range_str + '</CARDINAL> ')

    return input_str, output_str


def normalize_version(input_str):
    """
    Normalize many kinds of version, such as softwares, models, etc
    """
    return input_str

def norm_multiply_number(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    vi_numbers = re.findall(r'[0-9]*[.,]*[0-9]+\s*x\s*[0-9]*[.,]*[0-9]+\s*x\s*[0-9]*[.,]*[0-9]|[0-9]*[.,]*[0-9]+\s*x\s*[0-9]*[.,]*[0-9]', output_str)
    if len(vi_numbers) > 0:
        for vi_number in vi_numbers:
            vi_number_ver = multiply(vi_number)
            input_str = input_str.replace(vi_number, ' ' + vi_number + ' ')
            print(input_str)
            output_str = output_str.replace(vi_number, ' ' + vi_number_ver + ' ')
            print(output_str)

    return input_str, output_str


def normalize_number(input_str, output_str):
    input_str, output_str = norm_number_type_1(input_str, output_str)
    input_str, output_str = norm_number_type_2(input_str, output_str)
    input_str, output_str = norm_number_type_3(input_str, output_str)

    return input_str, output_str


def norm_number_type_1(input_str, output_str):
    """
    Normalize number
    """
    # Normalize vi-style numbers: '2.300 Euro', '25.320 vé', etc
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    vi_numbers = re.findall(r'\s([\d]+\.[\d]+\.*[\d]*\.*[\d]*\.*[\d]*)\s', output_str)
    if len(vi_numbers) > 0:
        for vi_number in vi_numbers:
            vi_number_norm = "".join(vi_number.split('.'))
            if int(vi_number_norm) >= 1000:
                input_str = input_str.replace(' ' + vi_number + ' ', ' ' + vi_number + ' ')
                output_str = output_str.replace(' ' + vi_number + ' ', ' ' + vi_number_norm + ' ')
            else:
                vi_number_ver = version2words(vi_number)
                input_str = input_str.replace(' ' + vi_number + ' ', ' ' + vi_number + ' ')
                output_str = output_str.replace(' ' + vi_number + ' ', ' ' + vi_number_ver + ' ')

    return input_str, output_str


def norm_number_type_2(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # Normalize numbers with comma format: '224,3 tỷ', '16,2 phần trăm', etc
    numbers_w_comma = re.findall(r'\s([\d]+,[\d]+)\s', output_str)
    if len(numbers_w_comma) > 0:
        for number in numbers_w_comma:
            number_str = num2words_float(number)
            # print(number, ":", number_str)
            input_str = input_str.replace(' ' + number + ' ',' ' + number + ' ')
            output_str = output_str.replace(' ' + number + ' ',' ' + number_str + ' ')

    return input_str, output_str


def norm_number_type_3(input_str, output_str):
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    numbers = re.findall(r'(\d+)', output_str)
    if len(numbers) > 0:
        for number in numbers:
            number_str = num2words_fixed(number)
            input_str = input_str.replace(' '+number +' ',' ' + number + ' ')
            output_str = output_str.replace(' '+number +' ',' ' + number_str + ' ')

    return input_str, output_str


def normalize_time(input_str, output_str):
    """
    Normalize time
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    # time_patterns = re.compile(r'\b(0?[0-9]|1\d|2[0-4])[:hg](0?[0-9]|[1-5]\d|)\b')
    time_patterns = re.compile(r'(\d+)(\:|h)(0?[0-9]|[1-5][0-9])(\:|p)([1-5][0-9]|0?[0-9])|(\d+)(\:|h)([1-5][0-9]|0?[0-9])|(\d+)h')
    temp_str_time = input_str
    times = []
    while(time_patterns.search(temp_str_time)):
        time = time_patterns.search(temp_str_time)
        times.append(time.group())
        temp_str_time = temp_str_time[time.span()[1]:]

    times = [time for time in times if not(time.startswith('24') and (time[3:]>'00')) ]
    if len(times) > 0:
        for time in times:
            time_str = time2words(time)
            #print('time:', time, '-', time_str)
            input_str = input_str.replace(' ' + time + ' ', ' <TIME>' + time + '</TIME> ')
            output_str = output_str.replace(' ' + time + ' ', ' <TIME>' + time_str + '</TIME> ')

    return input_str, output_str


def normalize_sport_score(input_str, output_str):
    """
    Normalize sport scores
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    scores = re.findall('\s[0-9]+\-[0-9]+\s', input_str)
    sport_ngrams = ['tỷ số', 'chiến thắng', 'trận đấu', 'tỉ số', 'bàn thắng', 'trên sân', 'đội bóng', 'thi đấu', 'cầu thủ',\
                    'vô địch', 'mùa giải', 'đánh bại', 'đối thủ', 'bóng đá', 'gỡ hòa', 'chung kết', 'bán kết', 'ghi bàn', \
                    'chủ nhà', 'tiền đạo', 'dứt điểm', 'tiền vệ', 'tiền đạo', 'thua']
    is_sport = 0
    for item in sport_ngrams:
        if (input_str.find(item)) != -1:
            is_sport = 1
            break
    if is_sport == 1 and len(scores) > 0:
        for score in scores:
            lscore = int(score.split('-')[0])
            rscore = int(score.split('-')[1])
            score_norm = num2words(lscore, lang='vi') + ' ' + num2words(rscore, lang='vi')
            input_str = input_str.replace(score, ' <CARDINAL>' + score + '</CARDINAL> ')
            output_str = output_str.replace(score, ' <CARDINAL>' + score_norm + '</CARDINAL> ')

    return input_str, output_str


def norm_fraction(fraction):
    first_num = fraction.split('/')[0]
    second_num = fraction.split('/')[1]
    if int(second_num) > 10:
        fraction_str = num2words_fixed(first_num) + ' trên ' + num2words_fixed(second_num)
    else:
        fraction_str = num2words_fixed(first_num) + ' phần ' + num2words_fixed(second_num)

    return fraction_str


def norm_tag_fraction(input_str, output_str):
    """
    Normalize number range.
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    p = re.compile(r"(thứ|hơn|gần|:|hạng|được|tới|góp|là|có|lên|bằng|[Cc]hiếm|giảm|tỷ lệ|tỉ lệ|khoảng)\s[0-9]+\s*\/\s*[0-9]+\s")
    temp_str = input_str
    ratio_list = []
    while(p.search(temp_str)):
        ratio = p.search(temp_str)
        x = ratio.group().replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
        ratio_list.append(x.split()[-1])
        temp_str = temp_str[ratio.span()[1]-1:]

    ratio_list = [item for item in ratio_list if int(item.split('/')[0]) < int(item.split('/')[1])]

    if len(ratio_list) > 0:
        # print(str(ratio_list), ' : ', input_str)
        for ratio in ratio_list:
            first_num = ratio.split('/')[0]
            second_num = ratio.split('/')[1]
            if int(second_num) > 10:
                ratio_str = num2words_fixed(first_num) + ' trên ' + num2words_fixed(second_num)
            else:
                ratio_str = num2words_fixed(first_num) + ' phần ' + num2words_fixed(second_num)

            input_str = input_str.replace(ratio, ' <FRACTION>' + ratio + '</FRACTION> ')
            output_str = output_str.replace(ratio, ' <FRACTION>' + ratio_str + '</FRACTION> ')

    return input_str, output_str


def normalize_email(input_str, output_str):
    """
    Normalize email addresses.
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    emails = re.findall('[a-zA-Z0-9]\S*@\S*[a-zA-Z0-9]', input_str)
    if len(emails) > 0:
        print(emails)

    return input_str, output_str


def normalize_url(input_str, output_str):
    """
    Normalize urls.
    """
    input_str = ' ' + input_str + ' '
    output_str = ' ' + output_str + ' '
    urls = re.findall(r"[a-zA-Z0-9]\S*\.com\S*|[a-zA-Z0-9]\S*\.net\S*|[a-zA-Z0-9]\S*\.org\S*|[a-zA-Z0-9]\S*\.vn\S*|[a-zA-Z0-9]\S*\.edu\S*|[a-zA-Z0-9]\S*\.gov\S*", input_str)
    urls = [item for item in urls if item.find('@')==-1]
    # Should add more top-level domain names: .uk, .io, .jp, etc if necessary

    return input_str, output_str


def norm_vnmese_accent(str):
    words = str.split(' ')
    for i in range(len(words)):
        if len(words[i]) <= 3:
            if not words[i].startswith('qu'):
                words[i] = words[i].replace("uỳ", "ùy")
                words[i] = words[i].replace("uý", "úy")
                words[i] = words[i].replace("uỷ", "ủy")
                words[i] = words[i].replace("uỹ", "ũy")
                words[i] = words[i].replace("uỵ", "ụy")
            else:
                words[i] = words[i].replace("ùy", "uỳ")
                words[i] = words[i].replace("úy", "uý")
                words[i] = words[i].replace("ủy", "uỷ")
                words[i] = words[i].replace("ũy", "uỹ")
                words[i] = words[i].replace("ụy", "uỵ")

            words[i] = words[i].replace("oà", "òa")
            words[i] = words[i].replace("oá", "óa")
            words[i] = words[i].replace("oả", "ỏa")
            words[i] = words[i].replace("oã", "õa")
            words[i] = words[i].replace("oạ", "ọa")
            words[i] = words[i].replace("oè", "òe")
            words[i] = words[i].replace("oé", "óe")
            words[i] = words[i].replace("oẻ", "ỏe")
            words[i] = words[i].replace("oẽ", "õe")
            words[i] = words[i].replace("oẹ", "ọe")
        else:
            words[i] = words[i].replace("òa", "oà")
            words[i] = words[i].replace("óa", "oá")
            words[i] = words[i].replace("ỏa", "oả")
            words[i] = words[i].replace("õa", "oã")
            words[i] = words[i].replace("ọa", "oạ")
            words[i] = words[i].replace("òe", "oè")
            words[i] = words[i].replace("óe", "oé")
            words[i] = words[i].replace("ỏe", "oẻ")
            words[i] = words[i].replace("õe", "oẽ")
            words[i] = words[i].replace("ọe", "oẹ")

    return ' '.join(words)


def read_foreign_words(f_foreign):
    df = pd.read_csv(f_foreign)
    words_foreign = df.word.values.tolist()
    trans_foreign = df.transcription.values.tolist()

    trans_dict = dict(zip(words_foreign, trans_foreign))

    return trans_dict


def read_abbre(f_abbre):
    fo = open(f_abbre, 'r')
    abbre_dict = dict()
    for line in fo:
        words = line.split('\t')
        abbre_dict[words[0]] = words[1].strip()

    return abbre_dict


read_abbre('Datasets/abbr.txt')


def norm_foreign_words(input_str, output_str, trans_dict):

    words_inp = input_str.split()
    words_out = output_str.split()

    for i in range(len(words_inp)):
        if words_inp[i] in trans_dict.keys():
            words_out[i] = '<FOREIGN>' + str(trans_dict[words_inp[i]]).replace('_',' ').strip() + '</FOREIGN>'
            words_inp[i] = '<FOREIGN>' + words_inp[i] + '</FOREIGN>'

    input_str = ' '.join(words_inp)
    output_str = ' '.join(words_out)

    return input_str, output_str


def norm_abbre(input_str, output_str, abbre_dict):

    words_inp = input_str.strip().split()
    words_out = output_str.strip().split()

    for i in range(len(words_inp)):
        if words_inp[i] in abbre_dict.keys():
#             words_out[i] = '<ABBRE>' + str(abbre_dict[words_inp[i]].replace(' ', '_')).strip() + '</ABBRE>'
#             words_inp[i] = '<ABBRE>' + words_inp[i] + '</ABBRE
            
            words_out[i] = str(abbre_dict[words_inp[i]].replace(' ', '_')).strip()
            words_inp[i] = words_inp[i]

    input_str = ' '.join(words_inp)
    output_str = ' '.join(words_out)

    return input_str, output_str


def norm_abbre_by_ngram():
    return False


def norm_code_type_1(input_str, output_str):
    # pattern = '(?=(\s[a-zA-Z]+[0-9]+\s|\s[0-9]+[a-zA-Z]+\s|\s[a-zA-Z]+[0-9]+[a-zA-Z]+\s))'
    pattern = '(?=(\s[a-zA-Z]+[0-9]+\s|\s[0-9]+[a-zA-Z]+\s|\s[a-zA-Z]+[0-9]+[a-zA-Z]+\s|\s[0-9]+[a-zA-Z]+[0-9]\s))'
    output_str = ' ' + output_str + ' '
    match_out = re.search(pattern, output_str)
    while match_out is not None:
        start_match_out = match_out.start(1)
        end_match_out = match_out.end(1)
        term_out = output_str[start_match_out:end_match_out]
        term_norm_out = term_out.replace(term_out, ' '.join(list(term_out)))
        term_norm_out = re.sub('(?<=\d)\s(?=\d)', '', term_norm_out) # remove space between number
        output_str = replace_str(output_str, start_match_out, end_match_out, term_norm_out, '', '')

        match_out = re.search(pattern, output_str)

    return input_str, output_str



def run(input_file, output_file, foreign_file, abbre_file):

    f_inp = open(input_file, 'r')
    fout_norm = open(output_file, "w")
    fout_norm_failure = open(input_file + '.fail', "w")

    writer_csv = csv.writer(fout_norm, delimiter='\t')
    writer_csv.writerow(['origin', 'written', 'spoken'])

    trans_dict = read_foreign_words(foreign_file)
    abbre_dict = read_abbre(abbre_file)

    count = 0

    for line in f_inp:
        count += 1
        print('\r count = %d' %count, end='\r')
        line = line.strip()
        input_line = line
        #    line = normalize_email(line)
        line = tokenize(line)
        line = ' '.join([i for i in re.split(r'([A-Za-z]+)', line) if i])
        line_inp, line_out = norm_abbre(line, line, abbre_dict)
        line_inp, line_out = norm_tag_verbatim(line_inp, line_out)
        line_inp, line_out = norm_punct(line_inp, line_out)
        line_inp, line_out = norm_foreign_words(line_inp, line_out, trans_dict=trans_dict)
        line_inp = line_inp.replace('_', ' ')
        line_out = line_out.replace('_', ' ')
        line_inp, line_out = normalize_AZ09(line_inp, line_out)
        line_inp, line_out = norm_tag_measure(line_inp, line_out)
        line_inp, line_out = norm_tag_fraction(line_inp, line_out)
        line_inp, line_out = normalize_date_range(line_inp, line_out)
        line_inp, line_out = normalize_date(line_inp, line_out)
        line_inp, line_out = normalize_time(line_inp, line_out)
        line_inp, line_out = normalize_phone_number(line_inp, line_out)
        line_inp, line_out = norm_digit(line_inp, line_out)
        line_inp, line_out = norm_tag_roman_num(line_inp, line_out)
        line_inp, line_out = normalize_number_range(line_inp, line_out)
        line_inp, line_out = normalize_sport_score(line_inp, line_out)
        line_inp, line_out = normalize_number(line_inp, line_out)
        line_inp, line_out = normalize_letters(line_inp, line_out)
        try:
            line_inp = norm_vnmese_accent(line_inp)
            line_out = norm_vnmese_accent(line_out)
        except:
            pass
        #    line = normalize_url(line)
        num = re.findall(r'\s\S*[0-9]+\S*\s', line_out)
        # if len(num) == 0:
            # fout_norm.write(input_line + '\t' + line + '\n')
        input_line = re.sub(' +',' ', input_line)
        line_inp = re.sub(' +',' ', line_inp)
        line_out = re.sub(' +',' ', line_out)
        writer_csv.writerow([input_line, line_inp, line_out])
        # else:
        #     fout_norm_failure.write(input_line + '\t' + line_inp + '\t' + line_out + '\n')

    fout_norm.close()
    fout_norm_failure.close()
    f_inp.close()


abbre_dict = read_abbre('Datasets/abbr.txt')
abbre_dict['cl'] = 'cờ lờ'
abbre_dict['Cl'] = 'Cờ lờ'
abbre_dict['clg'] = 'cờ lờ gờ'
abbre_dict['Clg'] = 'Cờ lờ gờ'
abbre_dict['clgt'] = 'cờ lờ gờ tờ'
abbre_dict['Clgt'] = 'Cờ lờ gờ tờ'
abbre_dict['clgv'] = 'cờ lờ gờ vờ'
abbre_dict['Clgv'] = 'Cờ lờ gờ vờ'
abbre_dict['cmn'] = 'cờ mờ nờ'
abbre_dict['Cmn'] = 'Cờ mờ nờ'
abbre_dict['cmnl'] = 'cờ mờ nờ lờ'
abbre_dict['Cmnl'] = 'Cờ mờ nờ lờ'
abbre_dict['dm'] = 'đờ mờ'
abbre_dict['Dm'] = 'Đờ mờ'
abbre_dict['đm'] = 'đờ mờ'
abbre_dict['Đm'] = 'Đờ mờ'
abbre_dict['dcm'] = 'đê cê mờ'
abbre_dict['đcm'] = 'đê cê mờ'
abbre_dict['Đcm'] = 'Đê cê mờ'
abbre_dict['dkm'] = 'đê ca mờ'
abbre_dict['đkm'] = 'đê ca mờ'
abbre_dict['dkmm'] = 'đê ca mờ mờ'
abbre_dict['Đkmm'] = 'Đê ca mờ mờ'
abbre_dict['đkmm'] = 'đê ca mờ mờ'
abbre_dict['Đkmm'] = 'Đê ca mờ mờ'
abbre_dict['vl'] = 'vờ lờ'
abbre_dict['Vl'] = 'Vờ lờ'
abbre_dict['vcl'] = 'vờ cờ lờ'
abbre_dict['Vcl'] = 'Vờ cờ lờ'
abbre_dict['vkl'] = 'vê ca lờ'
abbre_dict['Vkl'] = 'Vê ca lờ'
abbre_dict['qq'] = 'quần què'
abbre_dict['cmn'] = 'cờ mờ nờ'
abbre_dict['cmnr'] = 'cờ mờ nờ rờ'
abbre_dict['cc'] = 'cờ cờ'
abbre_dict['mn'] = 'mọi người'
abbre_dict['ms'] = 'mới'
abbre_dict['ns'] = 'nói'
abbre_dict['fb'] = 'ép bê'
abbre_dict['Fb'] = 'Ép bê'
abbre_dict['18+'] = 'mười tám cộng'
abbre_dict['pk'] = 'phải không'
abbre_dict['cmt'] = 'comment'
abbre_dict['Cmt'] = 'Comment'
abbre_dict["4'"] = 'phò'
abbre_dict['dc'] = 'được'
abbre_dict['đc'] = 'được'
abbre_dict['vđ'] = 'vãi đái'
abbre_dict['vs'] = 'với'
abbre_dict['trc'] = 'trước'
abbre_dict['Trc'] = 'Trước'
abbre_dict['đk'] = 'được'
abbre_dict['kk'] = 'ka ka'
abbre_dict['Kk'] = 'Ka ka'
abbre_dict['kkk'] = 'ka ka ka'
abbre_dict['Kkk'] = 'Ka ka ka'
abbre_dict['kkkk'] = 'ka ka ka ka'
abbre_dict['Kkkk'] = 'Ka ka ka ka'
abbre_dict['cx'] = 'cũng'
abbre_dict['m'] = 'mày'
abbre_dict['M'] = 'Mày'
abbre_dict['h'] = 'giờ'
abbre_dict['H'] = 'Giờ'
abbre_dict['b'] = 'bạn'
abbre_dict['B'] = 'Bạn'
abbre_dict['t'] = 'tao'
abbre_dict['T'] = 'Tao'
abbre_dict['k'] = 'không'
abbre_dict['ko'] = 'không'
abbre_dict['ntn'] = 'như thế nào'
abbre_dict['tdn'] = 'thế đéo nào'
abbre_dict['tđn'] = 'thế đéo nào'
abbre_dict['tsb'] = 'tiên sư bố'
abbre_dict['vc'] = 'vờ cờ' # vc, vợ chồng, việc
abbre_dict['Vc'] = 'Vờ cờ'
abbre_dict['vch'] = 'vãi chưởng'
abbre_dict['Vch'] = 'Vãi chưởng'
abbre_dict['đhs'] = 'đéo hiểu sao'
abbre_dict['Đhs'] = 'Đéo hiểu sao'
abbre_dict['cmm'] = 'con mẹ mày'
abbre_dict['cm'] = 'con mẹ'
abbre_dict['j'] = 'gì'
abbre_dict['ks'] = 'ca ét'
abbre_dict['bh'] = 'bây giờ'
abbre_dict['cdmm'] = 'cờ đờ mờ mờ'
abbre_dict['cđmm'] = 'cờ đờ mờ mờ'
# abbre_dict['nt'] = 'như thế' # nhắn tin
abbre_dict['bt'] = 'bình thường'
abbre_dict['3/'] = 'ba sọc'
abbre_dict['ss'] = 'so sánh'
abbre_dict['vk'] = 'vợ'
abbre_dict['Vk'] = 'Vợ'
abbre_dict['ck'] = 'chồng'
abbre_dict['Ck'] = 'Chồng'
abbre_dict['cdm'] = 'cộng đồng mạng'
abbre_dict['cđm'] = 'cộng đồng mạng'
abbre_dict['c'] = 'cờ'
abbre_dict['v'] = 'vậy'
abbre_dict['lm'] = 'làm'
abbre_dict['Lm'] = 'Làm'
abbre_dict['ngta'] = 'người ta'
abbre_dict['Ngta'] = 'Người ta'
abbre_dict['pls'] = 'please'
abbre_dict['Pls'] = 'Please'
abbre_dict['tk'] = 'tài khoản'
abbre_dict['dcd'] = 'đéo chịu được'
abbre_dict['đcđ'] = 'đéo chịu được'
abbre_dict['Dcd'] = 'Đéo chịu được'
abbre_dict['Đcđ'] = 'Đéo chịu được'
abbre_dict['l'] = 'lờ'
abbre_dict['vn'] = 'việt nam'
abbre_dict['Vn'] = 'Việt nam'
abbre_dict['VN'] = 'Việt Nam'
abbre_dict['ng'] = 'người'
abbre_dict['Ng'] = 'Người'
abbre_dict['s'] = 'sao'
abbre_dict['S'] = 'Sao'
abbre_dict['r'] = 'rồi'
abbre_dict['R'] = 'Rồi'
abbre_dict['đb'] = 'đê bê'
abbre_dict['ml'] = 'mờ lờ'
# abbre_dict['xl'] = 'xờ lờ' 
abbre_dict['cđg'] = 'cái đéo gì'
abbre_dict['clm'] = 'cờ lờ mờ'
abbre_dict['Clm'] = 'Cờ lờ mờ'
abbre_dict['hp'] = 'hạnh phúc'
abbre_dict['sdt'] = 'số điện thoại'
abbre_dict['sđt'] = 'số điện thoại'
abbre_dict['nv'] = 'như vậy'
abbre_dict['tg'] = 'thời gian'
abbre_dict['cg'] = 'cũng'
abbre_dict['nyc'] = 'người yêu cũ'
abbre_dict['Nyc'] = 'Người yêu cũ'
abbre_dict['ny'] = 'người yêu'
abbre_dict['mk'] = 'mình'
# abbre_dict['dk'] = 'được'
abbre_dict['bthg'] = 'bình thường'
abbre_dict['Bthg'] = 'Bình thường'
abbre_dict['klq'] = 'không liên quan'
abbre_dict['Klq'] = 'Không liên quan'
abbre_dict['kq'] = 'kết quả'
abbre_dict['Kq'] = 'Kết quả'
# abbre_dict['cs'] = 'cộng sản' 
# abbre_dict['Cs'] = 'Cộng sản'
abbre_dict['CS'] = 'Cộng Sản'
# abbre_dict['hđ'] = 'hiệp định' # 'hợp đồng'
# abbre_dict['Hđ'] = 'Hiệp định' # 'Hợp đồng'
abbre_dict['tq'] = 'trung quốc'
abbre_dict['Tq'] = 'Trung quốc'
abbre_dict['TQ'] = 'Trung Quốc'
abbre_dict['đ'] = 'đéo'
abbre_dict['NKYN'] = 'Nhật Ký Yêu Nước'
abbre_dict['ae'] = 'anh em'
abbre_dict['Ae'] = 'Anh em'
abbre_dict['ce'] = 'chị em'
abbre_dict['Ce'] = 'Chị em'
abbre_dict['glhf'] = 'good luck have fun'
abbre_dict['Glhf'] = 'Good luck have fun'
abbre_dict['ph'] = 'phút'
abbre_dict['ctrl'] = 'control'
abbre_dict['Ctrl'] = 'Control'
abbre_dict['dk'] = 'được' # điều kiện
abbre_dict['kg'] = 'không' # kg
abbre_dict['btrai'] = 'bạn trai' 
abbre_dict['gđ'] = 'gia đình' 
# abbre_dict['nc'] = 'nói chuyện' # nước
abbre_dict['hsd'] = 'hạn sử dụng' 
abbre_dict['IGL'] = 'in game leader' 
abbre_dict['qc'] = 'quảng cáo' 
abbre_dict['Qc'] = 'Quảng cáo' 
abbre_dict['Chs'] = 'Chả hiểu sao' 
abbre_dict['chs'] = 'chả hiểu sao' 
abbre_dict['pr'] = 'pê rờ' 
abbre_dict['Pr'] = 'Pê rờ' 
abbre_dict['cb'] = 'chuẩn bị' 
abbre_dict['hqa'] = 'hôm qua' 
abbre_dict['Hqa'] = 'Hôm qua'
abbre_dict['hqua'] = 'hôm qua' 
abbre_dict['Hqua'] = 'Hôm qua' 
abbre_dict['sml'] = 'sờ mờ lờ' 
abbre_dict['Sml'] = 'Sờ mờ lờ' 
abbre_dict['kh'] = 'không' 
abbre_dict['bn'] = 'bao nhiêu' #bạn
abbre_dict['tmv'] = 'thẩm mỹ viện' 
abbre_dict['sd'] = 'sử dụng' 
abbre_dict['vcd'] = 'vãi cả đái' 
abbre_dict['Vcd'] = 'Vãi cả đái' 
abbre_dict['vcđ'] = 'vãi cả đái' 
abbre_dict['Vcđ'] = 'Vãi cả đái' 
abbre_dict['ktra'] = 'kiểm tra' 
abbre_dict['ctv'] = 'cộng tác viên' 
abbre_dict['sx'] = 'sản xuất' 
abbre_dict['nta'] = 'người ta' 
abbre_dict['Nta'] = 'Người ta' 
abbre_dict['dt'] = 'dễ thương'  # điện thoại
abbre_dict['cmj'] = 'con mẹ gì' 
abbre_dict['Cmj'] = 'Con mẹ gì'  
abbre_dict['lsao'] = 'làm sao'  
abbre_dict['bth'] = 'bình thường'  
abbre_dict['Bth'] = 'Bình thường'  
abbre_dict['đt'] = 'điện thoại'  
abbre_dict['Đt'] = 'Điện thoại'  
abbre_dict['qt'] = 'quan tâm'  
abbre_dict['Qt'] = 'Quan tâm'  
abbre_dict['4 `'] = 'phò'  
abbre_dict['mv'] = 'em vi'  
abbre_dict['Clb'] = 'Câu lạc bộ'  
abbre_dict['clb'] = 'câu lạc bộ'  
abbre_dict['sk'] = 'sức khỏe'  
abbre_dict['kcn'] = 'khu công nghiệp' 
abbre_dict['bds'] = 'bất động sản'  
abbre_dict['bđs'] = 'bất động sản'  
abbre_dict['cty'] = 'công ty'  
abbre_dict['xklđ'] = 'xuất khẩu lao động'  
abbre_dict['XKLĐ'] = 'xuất khẩu lao động'  
abbre_dict['///'] = 'sọc'  
abbre_dict['kd'] = 'kinh doanh'  
abbre_dict['nđt'] = 'nhà đầu tư'  
abbre_dict['Nđt'] = 'Nhà đầu tư'  
abbre_dict['tgian'] = 'thời gian'  
abbre_dict['hsau'] = 'hôm sau'  
abbre_dict['hnay'] = 'hôm nay'  
abbre_dict['vlon'] = 'vãi lol'  
# abbre_dict['nt'] = 'nhắn tin' #như thế 
abbre_dict['bm'] = 'bố mẹ'  
abbre_dict['qđ'] = 'quyết định'  
abbre_dict['cf'] = 'cà phê'  
abbre_dict['vlin'] = 'vãi lìn'  
abbre_dict['gr'] = 'group'  
abbre_dict['vcc'] = 'vãi cả cờ'  
abbre_dict['Vcc'] = 'Vãi cả cờ'  
abbre_dict['th'] = 'thằng'  
abbre_dict['Th'] = 'Thằng'  
abbre_dict['md'] = 'mất dạy'  
abbre_dict['bl'] = 'bình luận'
abbre_dict['blv'] = 'bình luận viên'  
# abbre_dict['sp'] = 'sản phẩm' #support  
# abbre_dict['xh'] = 'xã hội' #xuất hiện 
abbre_dict['gd'] = 'gia đình'  
abbre_dict['hk'] = 'không'  
abbre_dict['hok'] = 'không'  
abbre_dict['rp'] = 'report'  
abbre_dict['ib'] = 'inbox'  
abbre_dict['bxh'] = 'bảng xêp hạng'  
# abbre_dict['tr'] = 'trời'  
# abbre_dict['Tr'] = 'Trời'  
abbre_dict['rr'] = 'rẻ rách'  
abbre_dict['đbrr'] = 'đầu bờ rẻ rách'  
abbre_dict['mng'] = 'mọi người'  
abbre_dict['dklm'] = 'đê ka lờ mờ'  
abbre_dict['ysl'] = 'yếu sinh lý'  
abbre_dict['xxx'] = 'x x x'  
abbre_dict['BCS'] = 'Bao Cao Su'  
abbre_dict['dlv'] = 'dư luận viên'  
abbre_dict['VC'] = 'Việt Cộng'  
abbre_dict['DMCS'] = 'Đờ mờ cờ sờ'  
abbre_dict['DMCS'] = 'Đờ mờ cờ sờ'  
abbre_dict['atsm'] = 'ảo tưởng sức mạnh'  
abbre_dict['cocc'] = 'cê âu cê cê'  
abbre_dict['ahbp'] = 'anh hùng bàn phím'  
abbre_dict['bmr'] = 'bỏ mẹ r'  
abbre_dict['cmnd'] = 'chứng minh nhân dân'  
abbre_dict['Cmnd'] = 'Chứng minh nhân dân'  
abbre_dict['cccd'] = 'căn cước công dân'  
abbre_dict['Cccd'] = 'Căn cước công dân'  
abbre_dict['zl'] = 'zậy luôn'  

del abbre_dict['HL']  # Hà Lan
del abbre_dict['GĐ']  # Gia đình
del abbre_dict['CV']  # CV, công việc
del abbre_dict['TTNN']
del abbre_dict['CS']  # Cộng Sản, Cuộc sống, Computer Science
del abbre_dict['TNS']
del abbre_dict['TT']  # Tổng thống
del abbre_dict['HNV'] # Hạ nghị viện
del abbre_dict['TNV'] # thượng nghị viện
del abbre_dict['CT'] # counter-terrorist
del abbre_dict['TG'] 
del abbre_dict['NN'] 
del abbre_dict['CP'] #Chipu

def norm_sentence(line):
    line = line.strip()
    input_line = line
    #    line = normalize_email(line)
    line = tokenize(line)
    line_inp, line_out = normalize_09unit(line, line)
    line_inp, line_out = normalize_decimal(line_inp, line_out)
    line_inp, line_out = normalize_dottedwords(line_inp, line_out)
    line_inp, line_out = normalize_datetime(line_inp, line_out)
    line_inp, line_out = norm_punct(line_inp, line_out)
    line_out = ' '.join([i for i in re.split(r'(\d+)', line_out) if i])
    line_inp, line_out = norm_abbre(line_out, line_out, abbre_dict)
    line_inp, line_out = norm_tag_verbatim(line_inp, line_out)
    line_inp, line_out = normalize_decimal(line_inp, line_out)
#     line_inp, line_out = norm_foreign_words(line_inp, line_out, trans_dict=trans_dict)
    line_inp = line_inp.replace('_', ' ')
    line_out = line_out.replace('_', ' ')
    line_inp, line_out = normalize_AZ09(line_inp, line_out)
    line_inp, line_out = norm_tag_measure(line_inp, line_out)
    line_inp, line_out = norm_tag_fraction(line_inp, line_out)
    line_inp, line_out = normalize_date_range(line_inp, line_out)
    line_inp, line_out = normalize_date(line_inp, line_out)
    line_inp, line_out = normalize_time(line_inp, line_out)
    line_inp, line_out = normalize_phone_number(line_inp, line_out)
    line_inp, line_out = norm_digit(line_inp, line_out)
    line_inp, line_out = norm_tag_roman_num(line_inp, line_out)
    line_inp, line_out = normalize_number_range(line_inp, line_out)
    line_inp, line_out = normalize_sport_score(line_inp, line_out)
    line_inp, line_out = normalize_number(line_inp, line_out)
    line_inp, line_out = normalize_letters(line_inp, line_out)
    try:
        line_inp = norm_vnmese_accent(line_inp)
        line_out = norm_vnmese_accent(line_out)
    except:
        pass
    line_out = remove_tag(line_out)
    return line_out

if __name__ == "__main__":
    a = norm_sentence("2h").strip()
    print(a)
