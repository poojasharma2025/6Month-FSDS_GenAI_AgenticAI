import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector

# Function to create a database connection
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='1234',
        database='EmpRegistration'
    )

# Function to add student to the database
def add_student():
    studentname = e2.get()
    coursename = e3.get()
    fee = e4.get()

    if not studentname or not coursename or not fee:
        messagebox.showerror("Input Error", "All fields must be filled.")
        return

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = "INSERT INTO registration (name, course, fee) VALUES (%s, %s, %s)"
        cursor.execute(sql, (studentname, coursename, fee))
        conn.commit()

        messagebox.showinfo("Success", "Student record added successfully!")
        e2.delete(0, tk.END)
        e3.delete(0, tk.END)
        e4.delete(0, tk.END)
        load_students()

    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Failed to insert student: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to update student record
def update_student():
    selected_item = listBox.selection()
    if not selected_item:
        messagebox.showerror("Selection Error", "Please select a student to update.")
        return

    studentid = e1.get()
    studentname = e2.get()
    coursename = e3.get()
    fee = e4.get()

    if not studentname or not coursename or not fee:
        messagebox.showerror("Input Error", "All fields must be filled.")
        return

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = "UPDATE registration SET name=%s, course=%s, fee=%s WHERE id=%s"
        cursor.execute(sql, (studentname, coursename, fee, studentid))
        conn.commit()

        messagebox.showinfo("Success", "Student record updated successfully!")
        e1.delete(0, tk.END)
        e2.delete(0, tk.END)
        e3.delete(0, tk.END)
        e4.delete(0, tk.END)
        load_students()

    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Failed to update student: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to delete a student
def delete_student():
    studentid = e1.get()
    if not studentid:
        messagebox.showerror("Selection Error", "Please select a student to delete.")
        return

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM registration WHERE id=%s", (studentid,))
        conn.commit()

        messagebox.showinfo("Success", "Student record deleted successfully!")
        e1.delete(0, tk.END)
        e2.delete(0, tk.END)
        e3.delete(0, tk.END)
        e4.delete(0, tk.END)
        load_students()

    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Failed to delete student: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to load students into the Treeview
def load_students():
    for row in listBox.get_children():
        listBox.delete(row)

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM registration")
        rows = cursor.fetchall()

        for row in rows:
            listBox.insert("", "end", values=row)

    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Failed to load students: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to populate the entry fields when a row is selected
def on_treeview_select(event):
    selected_item = listBox.selection()
    if selected_item:
        student = listBox.item(selected_item)
        studentid, studentname, coursename, fee = student['values']

        e1.config(state="normal")
        e1.delete(0, tk.END)
        e1.insert(0, studentid)

        e2.delete(0, tk.END)
        e2.insert(0, studentname)

        e3.delete(0, tk.END)
        e3.insert(0, coursename)

        e4.delete(0, tk.END)
        e4.insert(0, fee)

# GUI Setup
root = tk.Tk()
root.geometry('600x500')
root.title("Student Registration System")

tk.Label(root, text="Student ID").grid(row=0, column=0, padx=10, pady=10)
tk.Label(root, text="Name").grid(row=1, column=0, padx=10, pady=10)
tk.Label(root, text="Course").grid(row=2, column=0, padx=10, pady=10)
tk.Label(root, text="Fee").grid(row=3, column=0, padx=10, pady=10)

e1 = tk.Entry(root)
e1.grid(row=0, column=1, padx=10, pady=10)
e1.config(state="disabled")

e2 = tk.Entry(root)
e2.grid(row=1, column=1, padx=10, pady=10)

e3 = tk.Entry(root)
e3.grid(row=2, column=1, padx=10, pady=10)

e4 = tk.Entry(root)
e4.grid(row=3, column=1, padx=10, pady=10)

tk.Button(root, text="Add", command=add_student).grid(row=4, column=0, padx=10, pady=10)
tk.Button(root, text="Update", command=update_student).grid(row=4, column=1, padx=10, pady=10)
tk.Button(root, text="Delete", command=delete_student).grid(row=4, column=2, padx=10, pady=10)

cols = ("id", "name", "course", "fee")
listBox = ttk.Treeview(root, columns=cols, show="headings")
listBox.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

for col in cols:
    listBox.heading(col, text=col)
    listBox.column(col, width=150)

listBox.bind("<ButtonRelease-1>", on_treeview_select)

load_students()
root.mainloop()
