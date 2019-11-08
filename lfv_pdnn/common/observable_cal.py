import math

def delta_phi(phi1, phi2):
  """Calculates delta phi = phi1 - phi2.
  If delta phi is out of range [-PI, PI], k * 2PI will be added to the result
  to make delta phi is in range [-PI, PI], k is integer

  Args:
    phi1: float, first phi value.
    phi2: float, second phi value.
  """
  dphi = phi1 - phi2
  while dphi > math.pi:
    dphi -= 2 * math.pi
  while dphi < -math.pi:
    dphi += 2 * math.pi
  return dphi

def delta_r(eta1, phi1, eta2, phi2):
  """Calculates delta R.

  Args:
    eta1: float, eta value of first object.
    phi1: float, phi value of first object.
    eta2: float, eta value of second object.
    phi2: float, eat value of second object.

  Returns:
    dr: float, calculated by sqrt(dphi * dphi + deta * deta).
  """
  dphi = delta_phi(phi1, phi2)
  deta = eta1 - eta2
  dr = math.sqrt(dphi * dphi + deta * deta)
  return dr