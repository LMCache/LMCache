// SPDX-License-Identifier: Apache-2.0

use super::{check_nvme_ioctl_result, placement_id_to_u16, prepare_iouring_write_buffer};

#[test]
fn check_nvme_ioctl_result_accepts_success() {
    assert!(check_nvme_ioctl_result(0, "NVMe ioctl failed").is_ok());
}

#[test]
fn check_nvme_ioctl_result_rejects_nvme_status() {
    assert!(check_nvme_ioctl_result(1, "NVMe ioctl failed").is_err());
}

#[test]
fn placement_id_to_u16_accepts_valid_bounds() {
    assert_eq!(placement_id_to_u16(1).unwrap(), 1);
    assert_eq!(placement_id_to_u16(65535).unwrap(), 65535);
}

#[test]
fn placement_id_to_u16_rejects_reserved_and_out_of_range_values() {
    assert!(placement_id_to_u16(0).is_err());
    assert!(placement_id_to_u16(-1).is_err());
    assert!(placement_id_to_u16(65536).is_err());
}

#[test]
fn prepare_iouring_write_buffer_keeps_fixed_buffer_for_zero_tail() {
    let mut buf = vec![0u8; 4096];
    buf[..4].copy_from_slice(b"data");
    let ptr = buf.as_ptr() as usize;

    let prepared =
        prepare_iouring_write_buffer(ptr, 4096, 4, 4096, false, 4096, Some(3), false).unwrap();

    assert_eq!(prepared.ptr_addr, ptr);
    assert!(prepared.bounce.is_none());
    assert_eq!(prepared.fixed_buffer_idx, Some(3));
}

#[test]
fn padded_write_bounces_only_tail_without_modifying_source() {
    // Inspect the preparation helper to assert allocation size, which a
    // device round trip cannot distinguish from a full-buffer copy.
    let source = super::AlignedBuf::new(12288, 4096).unwrap();
    unsafe { std::ptr::write_bytes(source.as_mut_ptr(), 0xa5, 12288) };
    let ptr = source.as_ptr() as usize;
    let prepared =
        prepare_iouring_write_buffer(ptr, 10000, 10000, 12288, true, 4096, None, false).unwrap();
    assert_eq!(prepared.bounce.as_ref().unwrap().len, 4096);
    let vectors = prepared.iovecs.as_ref().unwrap();
    assert_eq!(vectors[0].base_addr, ptr);
    assert_eq!(vectors[0].len, 8192);
    assert_eq!(vectors[1].len, 4096);
    let tail =
        unsafe { std::slice::from_raw_parts(prepared.bounce.as_ref().unwrap().as_ptr(), 4096) };
    assert!(tail[..1808].iter().all(|v| *v == 0xa5));
    assert!(tail[1808..].iter().all(|v| *v == 0));
    assert!(
        unsafe { std::slice::from_raw_parts(source.as_ptr(), 12288) }
            .iter()
            .all(|v| *v == 0xa5)
    );
}

#[test]
fn padded_read_prepares_tail_copyback() {
    let target = super::AlignedBuf::new(12288, 4096).unwrap();
    let ptr = target.as_mut_ptr() as usize;
    let prepared =
        super::prepare_iouring_read_buffer(ptr, 10000, 10000, 12288, true, false, 4096, None)
            .unwrap();
    assert_eq!(prepared.bounce.as_ref().unwrap().len, 4096);
    assert_eq!(prepared.original_ptr, Some(ptr + 8192));
    assert_eq!(prepared.payload_len, Some(1808));
    assert_eq!(prepared.iovecs.as_ref().unwrap()[0].len, 8192);
}

#[test]
fn tail_preparation_preserves_fallbacks_and_aligned_fast_path() {
    let source = super::AlignedBuf::new(16384, 4096).unwrap();
    unsafe { std::ptr::write_bytes(source.as_mut_ptr(), 7, 16384) };
    let ptr = source.as_ptr() as usize;
    for size in [17, 4096, 4097, 8191, 8192, 10000] {
        let total = super::round_up(size, 4096);
        let prepared =
            prepare_iouring_write_buffer(ptr, size, size, total, true, 4096, Some(3), false)
                .unwrap();
        if size == total {
            assert!(prepared.bounce.is_none());
            assert!(prepared.iovecs.is_none());
            assert_eq!(prepared.fixed_buffer_idx, Some(3));
        } else {
            assert_eq!(prepared.bounce.as_ref().unwrap().len, 4096);
            assert_eq!(prepared.iovecs.is_some(), size > 4096);
            assert_eq!(prepared.fixed_buffer_idx, None);
        }
    }
    for (address, total, cmd) in [
        (ptr + 1, 12288, false),
        (ptr, 16384, false),
        (ptr, 12288, true),
    ] {
        let prepared =
            prepare_iouring_write_buffer(address, 10000, 10000, total, true, 4096, None, cmd)
                .unwrap();
        assert!(prepared.iovecs.is_none());
        assert_eq!(prepared.bounce.as_ref().unwrap().len, total);
    }
}

#[test]
fn vectored_completion_copies_tail_only_on_success() {
    for is_write in [false, true] {
        for result in [-libc::EIO, 0, 8192, 12288] {
            let target = super::AlignedBuf::new(12288, 4096).unwrap();
            unsafe { std::ptr::write_bytes(target.as_mut_ptr(), 0xcc, 12288) };
            let ptr = target.as_ptr() as usize;
            let prepared = super::prepare_iouring_read_buffer(
                ptr, 10000, 10000, 12288, true, false, 4096, None,
            )
            .unwrap();
            unsafe {
                std::ptr::write_bytes(prepared.bounce.as_ref().unwrap().as_mut_ptr(), 7, 4096)
            };
            let weak = std::sync::Arc::downgrade(prepared.bounce.as_ref().unwrap());
            let mut submission = super::IoSubmission {
                len: 12288,
                is_write,
                iovecs: prepared.iovecs,
                bounce: prepared.bounce,
                original_ptr: prepared.original_ptr,
                payload_len: prepared.payload_len,
                ..Default::default()
            };
            let outcome = super::handle_completion_result(&mut submission, result, false);
            assert_eq!(outcome.is_ok(), result == 12288);
            let bytes = unsafe { std::slice::from_raw_parts(target.as_ptr(), 12288) };
            let expected = if !is_write && result == 12288 {
                7
            } else {
                0xcc
            };
            assert!(bytes[8192..10000].iter().all(|value| *value == expected));
            assert!(bytes[10000..].iter().all(|value| *value == 0xcc));
            assert!(weak.upgrade().is_none());
        }
    }
}
