# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_executable(test_radio_health test_radio_health.c)
target_link_libraries(test_radio_health PRIVATE radio_health Threads::Threads)
add_dependencies(tests test_radio_health)
add_test(NAME test_radio_health
  COMMAND ./test_radio_health)

add_executable(test_radio_gain test_radio_gain.c)
target_link_libraries(test_radio_gain PRIVATE radio_gain Threads::Threads m)
add_dependencies(tests test_radio_gain)
add_test(NAME test_radio_gain
  COMMAND ./test_radio_gain)

add_executable(test_radio_gain_policy test_radio_gain_policy.c)
target_link_libraries(test_radio_gain_policy PRIVATE radio_gain Threads::Threads m)
add_dependencies(tests test_radio_gain_policy)
add_test(NAME test_radio_gain_policy
  COMMAND ./test_radio_gain_policy)

add_executable(test_radio_gain_samples test_radio_gain_samples.c)
target_link_libraries(test_radio_gain_samples PRIVATE radio_gain Threads::Threads)
add_dependencies(tests test_radio_gain_samples)
add_test(NAME test_radio_gain_samples COMMAND test_radio_gain_samples)

# Use the production binding with fake configuration/log/backend boundaries.
# radio_common's transitive loader/LOG libraries would also define those fakes.
add_executable(test_radio_gain_device test_radio_gain_device.c ../radio_gain_device.c)
target_link_libraries(test_radio_gain_device PRIVATE radio_gain Threads::Threads m)
# Keep the fake log boundary independent from a build-wide T tracer selection.
target_compile_options(test_radio_gain_device PRIVATE -UT_TRACER)
add_dependencies(tests test_radio_gain_device)
add_test(NAME test_radio_gain_device COMMAND test_radio_gain_device)

add_executable(test_radio_tx_power test_radio_tx_power.c)
target_link_libraries(test_radio_tx_power PRIVATE radio_gain Threads::Threads m)
add_dependencies(tests test_radio_tx_power)
add_test(NAME test_radio_tx_power COMMAND test_radio_tx_power)
