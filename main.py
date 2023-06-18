import os.path
import dbinit
import tkinter as tk
from screeninfo import get_monitors
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import knn

if not os.path.isdir("saved"):
    os.makedirs("saved")

if not os.path.isfile("clas.sqlite"):
    dbinit.initialize_db()


screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height


def new_dataset_window():

    def headers_write():

        def import_file():
            headers = []
            for e in entry_list:
                headers.append(e.get())

            name = name_entry.get()

            filename = file_entry.get()

            divider = divider_entry.get()

            cls = class_entry.get()

            dbinit.import_from_file(headers,name,filename,divider,cls)
            vals = dbinit.get_tables_names()
            select_data_combo.configure(values=vals)
            nd_window.destroy()

        num = dbinit.num_of_col_in_file(file_entry.get(),divider_entry.get())

        for i in range(num):
            entry_list.append(ttk.Entry(nd_window))
            entry_list[i].pack()

        pos = ["First", "Last"]

        class_entry = ttk.Combobox(nd_window, values=pos)
        class_entry.state(["readonly"])
        class_entry.pack()

        cr_button = ttk.Button(nd_window,text="IMPORT DATA",command=import_file)
        cr_button.pack()

    nd_window = tk.Toplevel(root)

    entry_list = []

    name_label = ttk.Label(nd_window,text="table name:")
    name_label.pack()

    name_entry = ttk.Entry(nd_window)
    name_entry.pack()

    file_label = ttk.Label(nd_window,text="file path: ")
    file_label.pack()

    file_entry = ttk.Entry(nd_window)
    file_entry.pack()

    divider_label = ttk.Label(nd_window, text="give divider: ")
    divider_label.pack()

    divider_entry = ttk.Entry(nd_window)
    divider_entry.pack()

    f_button = ttk.Button(nd_window,text="FILE GIVEN",command=headers_write)
    f_button.pack()


def save_to_database():
    if user_entry_output.get() == "":
        return
    vec = user_vector_input.get()
    cl = user_entry_output.get()
    picked = select_data_combo.get()
    dbinit.save_to_tabel(picked,vec,cl)
    pick_set("<<ComboboxSelected>>")


def clasify():
    inputt = user_vector_input.get()
    vec = inputt.split(',')
    fv = []
    for v in vec:
        fv.append(float(v))
    mod = knn.Model()
    mod.model_from_saved(select_model_combo.get())
    user_entry_output.delete(0,"end")
    user_entry_output.insert(0,mod.pred([fv]))


def get_model_list():
    return os.listdir("saved/")


def create_new_model_window():
    picked = select_data_combo.get()

    def num_choice():
        i = checkbox_var.get()
        if i == 1:
            min_nei_entry.state(['!disabled'])
            max_nei_entry.state(['!disabled'])
            neinum_entry.state(['disabled'])
        else:
            min_nei_entry.state(['disabled'])
            max_nei_entry.state(['disabled'])
            neinum_entry.state(['!disabled'])

    def create():

        rnd = int(rnd_entry.get())
        ts = float(test_set_size_entry.get())
        classcol = class_col_pos_entry.get()
        if classcol == "First":
            col = 0
        else:
            col = -1
        norm = bool(normal_entry.get())
        neinum = 5
        if checkbox_var.get() == 0:
            neinum = int(neinum_entry.get())
        mod = knn.Model(rnd,ts,picked,col,norm,neinum)
        mod.partition_data()
        mod.new_model_train()
        if checkbox_var.get() == 1:
            minnei = int(min_nei_entry.get())
            maxnei = int(max_nei_entry.get())
            mod.cross_val(minnei,maxnei)
        filename = filename_entry.get()
        mod.save_model_to_file(filename)
        select_model_combo.configure(values=get_model_list())
        cnm_window.destroy()

    def dis_score():
        rnd = int(rnd_entry.get())
        ts = float(test_set_size_entry.get())
        classcol = class_col_pos_entry.get()
        if classcol == "First":
            col = 0
        else:
            col = -1
        norm = bool(normal_entry.get())
        neinum = 5
        if checkbox_var.get() == 0:
            neinum = int(neinum_entry.get())
        mod = knn.Model(rnd, ts, picked, col, norm, neinum)
        mod.partition_data()
        mod.new_model_train()
        if checkbox_var.get() == 1:
            minnei = int(min_nei_entry.get())
            maxnei = int(max_nei_entry.get())
            mod.cross_val(minnei, maxnei)
        sc = mod.score()
        score_lab.configure(text=f"Średnia dokładność: {sc.mean()}")

    if picked != "":
        cnm_window = tk.Toplevel(root)

        filename_label = ttk.Label(cnm_window, text="set save file name: ")
        filename_label.pack()
        filename_entry = ttk.Entry(cnm_window)
        filename_entry.pack()

        rnd_label = ttk.Label(cnm_window,text="set random seed: ")
        rnd_label.pack()
        rnd_entry = ttk.Spinbox(cnm_window,from_=0,to=9999,increment=10)
        rnd_entry.pack()

        test_set_size_label = ttk.Label(cnm_window, text="set test set size: ")
        test_set_size_label.pack()
        test_set_size_entry = ttk.Spinbox(cnm_window,from_=0.01,to=100,format="%.2f",increment=0.01)
        test_set_size_entry.pack()

        pos = ["First", "Last"]

        class_col_pos_label = ttk.Label(cnm_window, text="set class column position: ")
        class_col_pos_label.pack()
        class_col_pos_entry = ttk.Combobox(cnm_window, values=pos)
        class_col_pos_entry.state(["readonly"])
        class_col_pos_entry.pack()

        ifnorm = ["True", "False"]

        normal_label = ttk.Label(cnm_window, text="normalize? ")
        normal_label.pack()
        normal_entry = ttk.Combobox(cnm_window, values=ifnorm)
        normal_entry.state(["readonly"])
        normal_entry.pack()

        rows = dbinit.count_rows(picked)
        neinum_label = ttk.Label(cnm_window, text="set number of neighbours: ")
        neinum_label.pack()
        neinum_entry = ttk.Spinbox(cnm_window,from_=0,to=rows,increment=1)
        neinum_entry.pack()

        checkbox_var = tk.IntVar()
        check_crval = ttk.Checkbutton(cnm_window,text="Find best n",variable=checkbox_var, command=num_choice)
        check_crval.pack()

        min_nei_label = ttk.Label(cnm_window, text="set cross validation min neighbours: ")
        min_nei_label.pack()
        min_nei_entry = ttk.Spinbox(cnm_window,from_=0,to=rows,increment=1)
        min_nei_entry.state(["disabled"])
        min_nei_entry.pack()

        max_nei_label = ttk.Label(cnm_window, text="set cross validation max neighbours: ")
        max_nei_label.pack()
        max_nei_entry = ttk.Spinbox(cnm_window,from_=0,to=rows,increment=1)
        max_nei_entry.state(["disabled"])
        max_nei_entry.pack()

        score_button = ttk.Button(cnm_window, text="SCORE", command=dis_score)
        score_button.pack()

        score_lab = ttk.Label(cnm_window)
        score_lab.pack()

        cancel_button = ttk.Button(cnm_window,text="CANCEL",command=cnm_window.destroy)
        cancel_button.pack()

        create_button = ttk.Button(cnm_window,text="CREATE",command=create)
        create_button.pack()


def draw_chart(event):
    picked = select_data_combo.get()
    col1 = select_x_combo.get()
    col2 = select_y_combo.get()
    if col1 != "" and col2 != "" and col1 != col2:
        for w in chart_frame.winfo_children():
            if w.winfo_name() != "chart_op":
                w.destroy()
        df = dbinit.get_for_chart(col1,col2,picked)

        dfs = {}
        bc = df.groupby('class')
        for groups,data in bc:
            dfs[groups] = data
        figure3 = plt.Figure(figsize=(8, 4), dpi=60)
        ax3 = figure3.add_subplot(111)
        for d in dfs:
            ax3.scatter(dfs[d][col1], dfs[d][col2], label=d)
        scatter3 = FigureCanvasTkAgg(figure3, chart_frame)
        scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        ax3.set_xlabel(col1)
        ax3.set_ylabel(col2)
        ax3.legend()


def load_data(picked,treeview):
    data = dbinit.fetch_data(picked)
    treeview.delete(*treeview.get_children())
    for row in data:
        l = ()
        for v in range(len(row)):
            l += (row[v],)
        treeview.insert("","end",values=l)


def pick_set(event):
    picked = select_data_combo.get()
    select_x_combo.set("")
    select_y_combo.set("")
    for w in data_chart_frame.winfo_children():
        w.destroy()
    treeview = ttk.Treeview(data_chart_frame)
    treeview["columns"] = dbinit.get_columns_names(select_data_combo.get())
    treeview.column("#0", width=0)
    colnum = 0
    for col in treeview["columns"]:
        colnum += 1
    xval.clear()
    yval.clear()
    for col in treeview["columns"]:
        if col != "class":
            xval.append(col)
            yval.append(col)
        treeview.heading(col, text=col)
        treeview.column(col, width=int(data_chart_frame.winfo_width()/colnum))
    treeview.pack()
    select_x_combo.configure(values=xval)
    select_y_combo.configure(values=yval)
    data_chart_frame.pack()
    load_data(picked,treeview)


root = tk.Tk()

# =================================ROOT===============================

root.title("PPY project")


root.geometry(f"{int(screen_width / 2)}x{int(screen_height / 2)}")

# ===========================TOP FRAME================================

top_frame = ttk.Frame(root, borderwidth=4, relief="ridge", width=int(screen_width), height=int(screen_height) / 4)
top_frame.pack(side="top", padx=10, pady=10)
top_frame.pack_propagate(0)
# ===========================CHART FRAME===============================

chart_frame = ttk.Frame(top_frame, borderwidth=4, relief="ridge", width=int(screen_width / 4),
                        height=int(screen_height) / 4)
chart_frame.pack(side="left")
chart_frame.pack_propagate(0)

chart_op_frame = ttk.Frame(chart_frame, name="chart_op", width=int(screen_width / 20),
                        height=int(screen_height) / 4)
chart_op_frame.pack(side="right")
chart_op_frame.pack_propagate(0)

xval = []
yval = []

select_x_combo = ttk.Combobox(chart_op_frame, values=xval)
select_x_combo.state(["readonly"])
select_x_combo.pack(anchor="center", padx=10, pady=10)
select_x_combo.bind("<<ComboboxSelected>>", draw_chart)

select_y_combo = ttk.Combobox(chart_op_frame, values=yval)
select_y_combo.state(["readonly"])
select_y_combo.pack(anchor="center", padx=10, pady=10)
select_y_combo.bind("<<ComboboxSelected>>", draw_chart)

# ===========================BUTTONS FRAME==============================

buttons_frame = ttk.Frame(top_frame, width=int(screen_width), height=chart_frame["height"], borderwidth=4,
                          relief="ridge")
buttons_frame.grid_columnconfigure(0,weight=1)
buttons_frame.grid_rowconfigure(0,weight=1)
buttons_frame.pack(side="right")
buttons_frame.pack_propagate(0)

values = dbinit.get_tables_names()
dataset_label = ttk.Label(buttons_frame, text="Pick dataset")
dataset_label.grid(row=0,column=0)

select_data_combo = ttk.Combobox(buttons_frame, values=values)
select_data_combo.state(["readonly"])
select_data_combo.grid(row=0,column=1)
select_data_combo.bind("<<ComboboxSelected>>", pick_set)

select_model_label = ttk.Label(buttons_frame, text="Pick model")
select_model_label.grid(row=1,column=0)

s = get_model_list()
select_model_combo = ttk.Combobox(buttons_frame,values=s)
select_model_combo.state(['readonly'])
select_model_combo.grid(row=1,column=1)

create_model_button = ttk.Button(buttons_frame, text="NEW MODEL", command=create_new_model_window)
create_model_button.grid(row=1,column=2)

user_vector_input = ttk.Entry(buttons_frame)
user_vector_input.grid(row=2,column=1)

user_entry_output = ttk.Entry(buttons_frame)
user_entry_output.grid(row=2,column=2)

pred_button = ttk.Button(buttons_frame, text="CLASSIFY", command=clasify)
pred_button.grid(row=3,column=1)

add_to_database_button = ttk.Button(buttons_frame, text="SAVE RECORD TO DATABASE", command=save_to_database)
add_to_database_button.grid(row=3,column=2)

new_dataset_button = ttk.Button(buttons_frame, text="NEW DATASET",command=new_dataset_window)
new_dataset_button.grid(row=3,column=0)

# ===========================DATA CHART FRAME========================

data_chart_frame = ttk.Frame(root, borderwidth=4, relief="ridge", width=int(screen_width),
                                 height=int(screen_height))
data_chart_frame.pack(side="bottom", padx=10, pady=10)
data_chart_frame.pack_propagate(0)


root.resizable(width=False, height=False)
root.mainloop()
