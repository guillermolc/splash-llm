import Foundation

/// Process-tree helpers for engine stop/kill.
enum ProcTree {
/// Synchronously kills an engine process tree (python wrapper + native child).
static func killTreeSync(_ pid: Int32) {
    let kids = shellString("/usr/bin/pgrep", ["-P", String(pid)])
    let kidList = kids.split(whereSeparator: { $0 == " " || $0 == "\n" }).compactMap { Int32($0) }
    kidList.forEach { kill($0, SIGTERM) }
    kill(pid, SIGTERM)
    let deadline = Date().addingTimeInterval(6)
    while Date() < deadline, kill(pid, 0) == 0 {
        Thread.sleep(forTimeInterval: 0.2)
    }
    if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
    kidList.forEach { kill($0, SIGKILL) }
    // Backstop: the native binary as an un-reparented stray.
    runShell("/bin/pkill", ["-f", "build/splash serve-native"])
}

static func portBusy(_ port: Int) -> Bool {
    guard let url = URL(string: "http://127.0.0.1:\(port)/status") else { return false }
    var req = URLRequest(url: url)
    req.timeoutInterval = 0.5
    let sem = DispatchSemaphore(value: 0)
    var busy = false
    URLSession.shared.dataTask(with: req) { _, resp, _ in
        if (resp as? HTTPURLResponse)?.statusCode == 200 { busy = true }
        sem.signal()
    }.resume()
    _ = sem.wait(timeout: .now() + 1)
    return busy
}

/// Runs a helper command and returns its trimmed stdout (or "").
static func shellString(_ launchPath: String, _ args: [String]) -> String {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: launchPath)
    p.arguments = args
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = FileHandle.nullDevice
    guard (try? p.run()) != nil else { return "" }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
}

static func runShell(_ launchPath: String, _ args: [String]) {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: launchPath)
    p.arguments = args
    p.standardOutput = FileHandle.nullDevice
    p.standardError = FileHandle.nullDevice
    try? p.run()
    p.waitUntilExit()
}


}
