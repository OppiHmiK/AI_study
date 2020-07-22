# USAGE : python dataAugmentizer.py -x test.xml -r t -b 2

from bs4 import BeautifulSoup as bs
from collections import Counter
import argparse
import lxml
import os

ap = argparse.ArgumentParser()
ap.add_argument('-x', '--xml', required = True)
ap.add_argument('-r', '--record', required = False, default = 'T', help = 'Make records files?')
ap.add_argument('-v', '--version', type = int, required = False, default = 2, help = "What's your build image records version")
args = vars(ap.parse_args())

class main:

    def __init__(self, xml, records = True, ver = 2):
        try:
            self.xml = xml
            self.ver = ver

            with open(xml, 'rb') as f:
                self.strHelper('XML file parsing...')
                self.soup = bs(f, 'lxml')
                self.strHelper('XML file parsing complete!', helper = 's')
            
            self.strHelper('Label crawling ...')
            labels = self.soup.select('images > image > box > label')
            self.strHelper('Label crawling complete!', helper = 's')

            self.lbList = []
            for lb in range(len(labels)):
                self.lbList.append(labels[lb].text)

            self.strHelper('Label extracting...')
            self.labelExtractor = sorted(list(set(self.lbList)))
            self.strHelper('Label extracting complete!', helper = 's')

            if records == True:
                self.createConfig()
                self.createRec()

            else:
                self.createConfig()

        except KeyboardInterrupt:
            self.strHelper('Did you press ctrl + c?', helper = 's')

    def createConfig(self):

        if not os.path.isdir('./build/config'):
            os.mkdir('./build/config')
            
        pyFile = self.configHelper()
        self.strHelper('Config file creating...')
        pyFile += '\nCLASSES = {\n'

        for (idx, lb) in enumerate(self.labelExtractor):
            pyFile += "\t '%s' : %d, \n"%(lb, int(idx) + 1)
        pyFile += '}'

        with open('./build/config/dlib_xml_config.py', 'w') as txt:
            txt.write(pyFile)
        self.strHelper('Creating complete!', helper = 's')

    def createRec(self):
        self.strHelper('Records file creating...')

        try:
            
            if self.ver == 2:
                os.system('python ./build/build_image_records2.py')
                self.strHelper('Creating complete!', helper = 's')
            
            elif self.ver == 1:
                os.system('python ./build/build_image_records.py')
                self.strHelper('Creating complete!', helper = 's')

        except Exception as e:
            self.strHelper(e, helper = 'e')

    def strHelper(self, string, helper = 'w'):

        if helper == 'w':
            print(f'[Please Wait] : {string}')
        
        elif helper == 's':
            print(f'[System] : {string} \n')

        elif helper == 'e':
            print(f'[Error] : {string}')

    def configHelper(self):
        pyFile = 'import os\n\nBASE_PATH="./"\n\nTRAIN_XML = os.path.sep.join([BASE_PATH, "%s"])\nTEST_XML = os.path.sep.join([BASE_PATH, "%s"])\n\n'\
        'if not os.path.isdir(f"{BASE_PATH}/records"):\n\tos.mkdir(f"{BASE_PATH}/records")\n\n'\
        'TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])\nTEST_RECORD = os.path.sep.join([BASE_PATH, "records/testing.record"])\nCLASSES_FILE = os.path.sep.join([BASE_PATH, "records/classes.pbtxt"])\n'%(self.xml, self.xml)\

        return pyFile

if __name__ == '__main__':

    r = True if args['record'].upper() == 'T' else False
    main(args['xml'], records = r, ver = args['version'])
