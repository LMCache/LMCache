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

    let prepared = prepare_iouring_write_buffer(ptr, 4096, 4, 4096, false, 4096, Some(3)).unwrap();

    assert_eq!(prepared.ptr_addr, ptr);
    assert!(prepared.bounce.is_none());
    assert_eq!(prepared.fixed_buffer_idx, Some(3));
}
