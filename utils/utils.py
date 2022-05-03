def print_task_info(info: str):
    print(info)


if __name__ == "__main__":
    with open("crash1.txt", "w") as f:
        f.write("5\n")
        f.write("1\n")
        f.write("1\n")
        f.write("2\n")
        f.write("3\n")
        q_num = 6
        f.write(f"{q_num}\n")
        for i in range(q_num):
            f.write("2 3\n")
