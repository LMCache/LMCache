// SPDX-License-Identifier: Apache-2.0

use super::{
    build_fixed_buffer_regions, check_nvme_ioctl_result, placement_id_to_u16,
    resolve_fixed_buffer_idx, FixedBufferRegion, MAX_FIXED_BUFFER_REGION_SIZE,
};

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
fn resolve_fixed_buffer_idx_accepts_complete_subranges() {
    let regions = [FixedBufferRegion {
        start: 100,
        end: 200,
        index: 7,
    }];

    assert_eq!(resolve_fixed_buffer_idx(&regions, 100, 100), Some(7));
    assert_eq!(resolve_fixed_buffer_idx(&regions, 125, 50), Some(7));
    assert_eq!(resolve_fixed_buffer_idx(&regions, 199, 1), Some(7));
}

#[test]
fn resolve_fixed_buffer_idx_rejects_uncovered_and_invalid_ranges() {
    let regions = [FixedBufferRegion {
        start: 100,
        end: 200,
        index: 7,
    }];

    assert_eq!(resolve_fixed_buffer_idx(&regions, 99, 1), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, 200, 1), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, 150, 51), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, 150, 0), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, usize::MAX, 2), None);
}

#[test]
fn resolve_fixed_buffer_idx_rejects_cross_region_requests() {
    let regions = [
        FixedBufferRegion {
            start: 100,
            end: 200,
            index: 3,
        },
        FixedBufferRegion {
            start: 200,
            end: 300,
            index: 9,
        },
    ];

    assert_eq!(resolve_fixed_buffer_idx(&regions, 199, 2), None);
    assert_eq!(resolve_fixed_buffer_idx(&regions, 200, 1), Some(9));
}

#[test]
fn resolve_fixed_buffer_idx_preserves_original_index_after_sort() {
    let mut regions = vec![
        FixedBufferRegion {
            start: 200,
            end: 300,
            index: 0,
        },
        FixedBufferRegion {
            start: 100,
            end: 200,
            index: 1,
        },
    ];
    regions.sort_unstable_by_key(|region| region.start);

    assert_eq!(resolve_fixed_buffer_idx(&regions, 100, 1), Some(1));
    assert_eq!(resolve_fixed_buffer_idx(&regions, 200, 1), Some(0));
}

#[test]
fn build_fixed_buffer_regions_accepts_adjacent_unsorted_regions() {
    let regions = build_fixed_buffer_regions(&[200, 100], &[100, 100]).unwrap();

    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].start, 100);
    assert_eq!(regions[0].end, 200);
    assert_eq!(regions[0].index, 1);
    assert_eq!(regions[1].start, 200);
    assert_eq!(regions[1].end, 300);
    assert_eq!(regions[1].index, 0);
}

#[test]
fn build_fixed_buffer_regions_rejects_invalid_inputs() {
    assert!(build_fixed_buffer_regions(&[100], &[]).is_err());
    assert!(build_fixed_buffer_regions(&[], &[]).is_err());
    assert!(build_fixed_buffer_regions(&[0], &[1]).is_err());
    assert!(build_fixed_buffer_regions(&[100], &[0]).is_err());
    assert!(build_fixed_buffer_regions(&[100], &[MAX_FIXED_BUFFER_REGION_SIZE + 1]).is_err());
    assert!(build_fixed_buffer_regions(&[usize::MAX], &[2]).is_err());
    assert!(build_fixed_buffer_regions(&[100, 150], &[100, 100]).is_err());

    let too_many = usize::from(u16::MAX) + 2;
    assert!(build_fixed_buffer_regions(&vec![1; too_many], &vec![1; too_many]).is_err());
}
