from bs4 import BeautifulSoup as bs
from argparse import ArgumentParser
from imutils import paths
import lxml
import cv2
import os

ap = ArgumentParser()
ap.add_argument('-o', '--option', required = True)
ap.add_argument('-x', '--xml', required = True)

ap.add_argument('-fp', '--fix_imgPath', required = False, default = 'resize')
ap.add_argument('-op', '--origin_imgPath', required = False, default = 'imgs')

ap.add_argument('-pw', '--parallel_width', required = False, default = 470, type = int)
ap.add_argument('-ph', '--parallel_height', required = False, default = 210, type = int)

ap.add_argument('-n', '--name', required = False, default = 'adjust')
opt = ap.parse_args()

class main:
    
    def __init__(self, **kwargs):

        option = kwargs['option'].lower()
        xml, name = kwargs['xml'], kwargs['name']
        pw, ph = kwargs['pwidth'], kwargs['pheight']
        ori, fix = kwargs['originPath'], kwargs['resizePath']


        oriImgpath = list(sorted(paths.list_images(ori)))[0]
        fixImgpath = list(sorted(paths.list_images(fix)))[0]
        oriImg, fixImg = cv2.imread(oriImgpath), cv2.imread(fixImgpath)
        
        print(oriImgpath, fixImgpath)

        with open(xml, 'rb') as f:
            print('xml parsing')
            self.soup = bs(f, 'lxml')
            print('parsing complete')

        text = "<?xml version='1.0' encoding='ISO-8859-1'?>\n<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n<dataset>\n<name>imglab dataset</name>\n<comment>Created by imglab tool.</comment>\n<images>\n"
        for image in self.soup.select('images > image'):

            fileName = image['file']
            text += f'<image file = "{fileName}">\n'
            for box in image.select('box'):
                if option == 'a':
                    top, left = int(box['top']) - ph, int(box['left']) - pw
                    width, height = int(box['width']), int(box['height'])

                elif option == 'r':

                    (w, h), (rw, rh)= oriImg.shape[:2], fixImg.shape[:2]
    
                    ratio_w, ratio_h = rw / w, rh / h
                    print(ratio_w, ratio_h)

                    top, left = int(int(box['top']) * ratio_w), int(int(box['left']) * ratio_h)
                    width, height = int(int(box['width']) * ratio_w), int(int(box['height']) * ratio_h)

                    print(f'fix t : {top}, fix l : {left}')
                    print(f'fix w : {width}, fix h : {height}')

                lb = box.select('label')[0].text
                print('lb : ', lb)
                text += f'<box top = "{top}" left = "{left}" width = "{width}" height = "{height}">\n\t\t\t<label>{lb}</label>\n\t\t</box>\n\t'

            text += '</image>\n'
        text += '</images>\n</dataset>'
        
        name = f'{name}_a' if option == 'a' else f'{name}_r'
        with open(f'{name}.xml', 'w') as f:
            f.write(text)
        
        print('xml writing')
        

if __name__ == '__main__':
    main(xml = opt.xml, option = opt.option, resizePath = opt.fix_imgPath, originPath = opt.origin_imgPath,  pwidth = opt.parallel_width, pheight = opt.parallel_height, name = opt.name)