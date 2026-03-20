import Foundation
import Metal

/// Loads weights.bin into a single Metal buffer.
public struct WeightLoader {
    /// Load raw weight binary into a shared-mode Metal buffer.
    public static func load(from url: URL, device: MTLDevice) throws -> MTLBuffer {
        let data = try Data(contentsOf: url)
        guard let buffer = device.makeBuffer(
            bytes: (data as NSData).bytes,
            length: data.count,
            options: .storageModeShared
        ) else {
            throw WeightLoadError.bufferCreationFailed
        }
        buffer.label = "weights"
        return buffer
    }

    public enum WeightLoadError: Error {
        case bufferCreationFailed
    }
}
