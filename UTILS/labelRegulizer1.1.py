from bs4 import BeautifulSoup as bs
from collections import Counter
import argparse
import lxml
import os

ap = argparse.ArgumentParser()
ap.add_argument('-x', '--xml', required = True)
ap.add_argument('-n', '--name', required = False, default = 'regulizer')
ap.add_argument('-i', '--info', required = False, default = 'T')
args = vars(ap.parse_args())

class main:

    def __init__(self, xml, name, info = True):
        try:
            self.xml = xml
            self.name = name

            with open(self.xml, 'rb') as f:
                self.strHelper('XML file parsing...')
                self.soup = bs(f, 'lxml')
                self.strHelper('XML file parsing complete!\n', helper='s')

            boxes = self.soup.select('images > image > box')
            labels = self.soup.select('images > image > box > label')

            lbList = []
            for lb in range(len(labels)):
                lbList.append(labels[lb].text)

            self.strHelper('ratio calculating...')
            counter, ratio = self.ratioCal(lbList)
            self.strHelper('ratio calculating complete!\n', helper = 's')            

            if info == True:
                for idx in range(len(counter)):
                    self.strHelper(f'label name : {counter[idx][0]}, # of label : {counter[idx][1]}', helper = 'i')

                print('\n')
                self.xmlWriter(counter, ratio)

            else:
                self.xmlWriter(counter, ratio)

        except KeyboardInterrupt:
            self.strHelper('Did you press ctrl + c?', 'e')

        except Exception as e:
            self.strHelper(e, helper = 'e')

    def ratioCal(self, lis):
        counter = Counter(lis).most_common()
        most = counter[0][1]
        ratio = [round(most / counter[idx][1])  for idx in range(len(counter))]
        return counter, ratio

    def strHelper(self, msg, helper = 'w'):
        if helper == 'w':
            print(f'[Please Wait] : {msg}')

        elif helper == 's':
            print(f'[System] : {msg}')

        elif helper == 'e':
            print(f'[Error] : {msg}')

        elif helper == 'i':
            print(f'[Info] : {msg}')

    def xmlWriter(self, counter, ratio):
        self.strHelper('writing xml file...')
        result = {}
        for idx in range(len(counter)):
            result[counter[idx][0]] = ratio[idx]        

        string = "<?xml version='1.0' encoding='ISO-8859-1'?>\n<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n<dataset>\n<name>imglab dataset</name>\n<comment>Created by imglab tool.</comment>\n<images>\n"
        for image in self.soup.find_all("image"):
            file = image["file"]
            string += f"<image file='{file}'>\n"
            for box in image.select('image > box'):
                label = box.find("label").text
                for idx in range(result[label]):
                    string += f'{box}\n'
            string += "</image>\n"
        string += "</images>\n</dataset>"
        
        with open(f'{self.name}.xml', 'w') as f:
            f.write(string)
        self.strHelper('writing xml file complete!', helper = 's')

if __name__ == '__main__':
    info = True if args['info'].upper() == 'T' else False
    main(xml = args['xml'], name = args['name'], info = info)