# (OPTIONAL) Overrides the copywrite config schema version
# Default: 1
schema_version = 1

project {
  # (OPTIONAL) SPDX-compatible license identifier
  # Leave blank if you don't wish to license the project
  # Default: "MPL-2.0"
  license = "Apache-2.0"

  # (OPTIONAL) Represents the copyright holder used in all statements
  # Default: IBM Corp.
  copyright_holder = "Center for Digital Humanities, Princeton University"

  # (OPTIONAL) Represents the year that the project initially began
  # This is used as the starting year in copyright statements
  # If set and different from current year, headers will show: "copyright_year, year-2"
  # If set and same as year-2, headers will show: "copyright_year"
  # If not set (0), the tool will auto-detect from git history (first commit year)
  # If auto-detection fails, it will fallback to current year only
  # Default: 0 (auto-detect)
  # copyright_year = 0

	# (OPTIONAL) If true, ignore updating the first year (start year) in copyright ranges.
	# End-year logic remains unchanged.
	# Default: false
	# ignore_year1 = false

  # (OPTIONAL) A list of globs that should not have copyright/license headers.
  # Supports doublestar glob patterns for more flexibility in defining which
  # files or folders should be ignored
  header_ignore = [
    ".venv/**",
    # "vendor/**",
    # "**autogen**",
  ]
}
