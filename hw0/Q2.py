from PIL import Image
import sys
ori_im = Image.open(sys.argv[1])
new_pixels = [(r//2,g//2,b//2) for (r,g,b) in ori_im.getdata()]
new_im = Image.new(ori_im.mode, ori_im.size)
new_im.putdata(new_pixels)
new_im.save('Q2.png')
