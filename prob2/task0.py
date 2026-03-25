first_names = [
"Aarav","Vivaan","Aditya","Vihaan","Arjun","Reyansh","Ishaan","Kabir","Dhruv","Aryan",
"Kian","Ayaan","Yash","Dev","Ansh","Harsh","Kunal","Rahul","Rohan","Amit",
"Varun","Siddharth","Manav","Nikhil","Akash","Arnav","Laksh","Parth","Om","Veer",
"Aditi","Ananya","Anika","Anushka","Aarohi","Aadhya","Diya","Ira","Ishita","Kavya",
"Kiara","Kritika","Meera","Myra","Navya","Naina","Neha","Pooja","Priya","Riya",
"Avni","Bhavna","Chitra","Divya","Garima","Ishani","Jhanvi","Khushi","Lavanya","Mahi"
]

last_names = [
"Sharma","Verma","Gupta","Agarwal","Singh","Kumar","Patel","Yadav","Reddy","Iyer",
"Nair","Menon","Chatterjee","Banerjee","Mukherjee","Das","Joshi","Mehta","Bansal","Kapoor",
"Malhotra","Khanna","Saxena","Tiwari","Pandey","Tripathi","Kulkarni","Deshmukh","Shetty","Pillai"
]

names = []

for f in first_names:
    for l in last_names:
        names.append(f + " " + l)
        if len(names) == 1000:
            break
    if len(names) == 1000:
        break

with open("TrainingNames.txt", "w") as file:
    for name in names:
        file.write(name + "\n")

print("Generated", len(names), "names")