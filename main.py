from flask import Flask, render_template,request
from datetime import datetime
start = None
end = None
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html")
@app.route('/uploaded',methods=['GET','POST'])
def uploaded():
    if request.method == 'POST':
        f = request.files['file']
        f.filename="whatsapp.txt"
        f.save(f.filename)
        from analysis2 import p,neg,neu,off,dataInsights
        res=[]
        res.append(off)
        l = []
        l.append(p)
        l.append(neu)
        l.append(neg)
        # print(l)
        context = {'analysis': dataInsights()}
        return render_template('Analysis.html',text=l,context=context,res=res)
    # return render_template('Analysis.html',{'sent':l})
    # return render_template('Analysis.html')

@app.route('/range',methods=['GET','POST'])
def range():  
    # context = {}
    if request.method == 'POST':
        start = request.form['start-date']
        print(start)
        end= request.form['end-date']
        print(end)
        from analysis2 import df2,sentiment_analysis
        ans,p,neg,neu=sentiment_analysis(df2,start,end)
        print(ans)
        context={'analysis':ans}      
        l=[]
        l.append(p) 
        # l.append(neu)
        l.append(neg) 
        return render_template('range.html',text=l,context=context)
    return render_template('range.html')
        
    

if __name__=="__main__":
    app.run(debug=True)