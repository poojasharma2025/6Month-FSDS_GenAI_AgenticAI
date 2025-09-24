import mysql.connector
conn = mysql.connector.connect(host = 'localhost', user = 'root', password = '1234', database = 'pythonDB')
mycursor = conn.cursor()
sql = 'insert into student (name,branch,id) values(%s,%s,%s)'
val = [('john','cse','56'),('mike','IT','78'),('tylor','ME','80')]
mycursor.executemany(sql,val)
conn.commit()
print(mycursor.rowcount,'record inserted')
