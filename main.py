import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from sketch import sketch
from cartoonize import cartoon,hist,sharpening,bilateral,blackwhite,bright,HSVcolour,HLScolour,invert,denoise,morph,autocontrast,contraststreching
from cartoonize import RGBcolour,logarthemic,exposure,blurring,curve,greyscale,heatfilter,grabcut,colourswap,warpfilter,backgroundsubstraction,averagefilter
from cartoonize import highpassfilter,lowpassfilter,vignetefilter,fliprotationfilter,translation,scaling,contourdetection,bestfilter,blurdetection,wienerfilter
from cartoonize import instagramfilter,dehaze

app = Flask(__name__)
# img =''
global filename, current_file
filename = ''
current_file =''
# basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/img', methods=['POST'])
def upload_image():
	global filename
	if 'file' not in request.files:
		# flash('No file selected')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		# flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		# filename = file.filename
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
		# flash('Image successfully uploaded and displayed')
		return render_template('home.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)


@app.route('/operation', methods=['POST'])
def operation():
	global filename, current_file
	# print(filename)
	
	if "cartoonize" in request.form:
	        
		img = cv2.imread('static/uploads/'+filename)
		#default_value=0
		#img=request.form['name']
		#gamaValue=request.form.get('name',default_value)
		#print(gamaValue)

		prev_filename = filename
		name = filename.split('.')
		filename1 = name[0]+"_1."+name[1]
		current_file = filename1		
		img = cartoon(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename1),img)
    		
		
		return render_template('home.html', filename=prev_filename, filename_1 = filename1)		
	elif "sketch" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		#alpha = request.form['gamma']
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img = sketch(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "hist" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img = hist(img)
		
		
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "blurring" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img = blurring(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "sharpening" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =sharpening(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "bilateral" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =bilateral(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "blackwhite" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =blackwhite(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "bright" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =bright(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "RGBcolour" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =RGBcolour(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
		
	elif "HSVcolour" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =HSVcolour(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "HLScolour" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =HLScolour(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "invert" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =invert(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "denoise" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		#img=int(request.form['name'])
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =denoise(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "morph" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =morph(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
		
	elif "autocontrast" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =autocontrast(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "contraststreching" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =contraststreching(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "logarthemic" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =logarthemic(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "exposure" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =exposure(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "curve" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =curve(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "greyscale" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =greyscale(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "heatfilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =heatfilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "grabcut" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =grabcut(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "warpfilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =warpfilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "backgroundsubstraction" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =backgroundsubstraction(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	
	elif "averagefilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =averagefilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "highpassfilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =highpassfilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
		
	elif "lowpassfilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =lowpassfilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "vignetefilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =vignetefilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "fliprotationfilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =fliprotationfilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "translation" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =translation(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "scaling" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =scaling(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "contourdetection" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =contourdetection(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "bestfilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =bestfilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "blurdetection" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =blurdetection(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif " wienerfilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img = wienerfilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "motionblur" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =motionblur(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "instagramfilter" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =instagramfilter(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	elif "dehaze" in request.form:
		img = cv2.imread('static/uploads/'+filename)
		prev_filename = filename
		name = filename.split('.')
		filename2 = name[0]+"_2."+name[1]
		current_file = filename2
		img =dehaze(img)
		cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2),img,sz)
		return render_template('home.html', filename=prev_filename, filename_2 = filename2)
	
	else:
		return render_template('home.html')

@app.route('/display/<filename>')
def display_image(filename):
	# global filename
	img = cv2.imread('static/uploads/'+filename)
	ret, jpeg = cv2.imencode('.jpg', img)
    # os.remove('static/uploads/'+filename)
	return jpeg.tobytes()

@app.route('/return-files')
def return_files():
	global current_file
	if current_file == '':
		return render_template('home.html')
	else:
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], current_file)
		return send_file(file_path, as_attachment=True, attachment_filename=current_file)


if __name__ == '__main__':
    app.run(debug=True)
