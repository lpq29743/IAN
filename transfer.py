import os
import unicodedata
import xml.etree.ElementTree as ET
from errno import ENOENT


input_fname = 'data/laptop/train.xml'
output_fname = 'data/laptop/train.txt'

if not os.path.isfile(input_fname):
    raise IOError(ENOENT, 'Not an input file', input_fname)
    
with open(output_fname, 'w') as f:
    tree = ET.parse(input_fname)
    root = tree.getroot()
    sentence_num = 0
    aspect_num = 0
    for sentence in root.iter('sentence'):
        sentence_num = sentence_num + 1
        text = sentence.find('text').text
        for asp_terms in sentence.iter('Opinions'):
            for asp_term in asp_terms.findall('Opinion'):
                if asp_term.get('polarity') != 'conflict' and asp_term.get('target') != 'NULL':
                    aspect_num = aspect_num + 1
                    new_text = ''.join((text[:int(asp_term.get('from'))], 'aspect_term', text[int(asp_term.get('to')):]))
                    f.write('%s\n' % new_text.strip())
                    f.write('%s\n' % asp_term.get('target'))
                    f.write('%s\n' % asp_term.get('polarity'))
                    print("Read %s sentences %s aspects" % (sentence_num, aspect_num))
