# -*- coding:utf8 -*-
# 数据源标注系统

import flask
import pymongo
import bson.binary
from io import BytesIO
from PIL import Image
import glob
from flask import render_template, redirect, session
import base64
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import Required
from flask_bootstrap import Bootstrap

app = flask.Flask(__name__)
app.config['SECRET_KEY']='123'
app.debug = True
Bootstrap(app)
db = pymongo.MongoClient('localhost', 27017).test

class LabelForm(FlaskForm):
    label_value = StringField('label', validators=[Required()])
    submit = SubmitField('确定')

# 将图片以binary格式存入mongodb
def save_file(f, cnt):
    name = f.name.split('/')[-1].split('_')[1]
    type = f.name.split('/')[-1].split('_')[2].split('zoom')[0]
    print('Now saving %s - %s' % (name, type))
    content = BytesIO(f.read())
    try:
        mime = Image.open(content).format.lower()
    except IOError:
        flask.abort(400)
    c = dict(content=bson.binary.Binary(content.getvalue()),
                mime=mime,
                name=name,
                type=type,
                index=cnt
                )
    db.files.save(c)
    return c['_id']

# 根据ID返回图片
@app.route('/f/<id>')
def serve_file(id):
    try:
        f = db.files.find_one({'index': int(id)})
        print('Now checking %s - %s' % (f['name'], f['type']))
        return flask.Response(f['content'], mimetype='image/' + f['mime'])
        #pic = {'index':id, 'name':f['name'], 'type':f['type'], 'content':f['content']}
        #return render_template("index.html", pic=pic, base64=base64)
    except:
        flask.abort(404)

# 样本显示及标注页面
@app.route('/show/<id>', methods=['GET', 'POST'])
def show(id):
    form = LabelForm()
    all = db.files.find().count()
    f = db.files.find_one({'index': int(id)})
    #print('Now checking %s - %s' % (f['name'], f['type']))
    
    if 'OCR' not in f.keys():
        f['OCR'] = None

    form.label_value.data = f['OCR']
    session['name'] = f['name']
    session['type'] = f['type']

    if 'label' in f.keys():
        label = f['label']
    else:
        label = None

    pic = {'index':id, 'name':f['name'], 'type':f['type'], 'ocr':f['OCR'], 'label':label, 'all':all}
    raw = base64.b64encode(f['content'])
    return render_template("index.html", form=form, pic=pic, raw=raw.decode('utf8'))

# 更新样本标注值
@app.route('/submit/<id>', methods=['GET', 'POST'])
def submit(id):
    form = LabelForm()
    if not form.validate_on_submit():
        db.files.update_one({'name': session.get('name'), 'type':session.get('type')},{"$set": {"label":form.label_value.data}})
        import pdb
        #pdb.set_trace()
        new_id = int(id) + 1
        return redirect('/show/' + str(new_id))
    return redirect('/show/' + str(id))

# 从文件夹里批量上传待标注的样本文件
@app.route('/upload', methods=['POST'])
def upload():
    file_list = glob.glob('/Users/ethan/MyCodes/baidu_tensor/raw/*.jpg')
    cnt = 1
    length = len(file_list)
    for file in file_list:
        f = open(file, 'rb')
        save_file(f, cnt)
        cnt += 1
    #f = flask.request.files['uploaded_file']
    import pdb
    #pdb.set_trace()
    #fid = save_file(f)
    print('All %d pics added' % length)
    return flask.redirect('/')

# 样本上传首页
@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <body>
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='uploaded_file'>
        <input type='submit' value='Upload'>
    </form>
    '''

if __name__ == '__main__':
    app.run(port=7777)