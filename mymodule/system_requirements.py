import psutil
# Monitor system resources
def monitor_resources():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu}%")
    print(f"Memory Usage: {memory.percent}% ({memory.used / (1024**3):.2f} GB used)")
