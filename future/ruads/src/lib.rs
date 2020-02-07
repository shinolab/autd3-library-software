extern crate libc;
extern crate libloading;
#[macro_use]
extern crate lazy_static;

mod ads_error;
mod local_ethercat;
mod native_methods;
mod remote_ethercat;

pub use local_ethercat::LocalADSLink;
pub use remote_ethercat::RemoteADSLink;
