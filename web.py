import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import subprocess
UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print "uploaded file"+filename
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('upload.html')

@app.route('/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/show/<filename>')
def uploaded_file(filename):
	#filename = 'http://127.0.0.1:5000/' + filename
    #print "python join.py -i " + str(filename)
    #proc=subprocess.Popen("python join.py -i " + str(filename),shell=True)
    #streamdata=proc.communicate()[0]
    #rc=proc.returncode
    import join
    rc=join.run(str(filename))
    print "Hello"
    return "success"
    # return render_template('function.html',value=False)


if __name__ == '__main__':
    app.config["SECRET_KEY"] = "ITSASECRET"
    app.run(port=8000,debug=True)


