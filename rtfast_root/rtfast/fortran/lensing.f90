
!Following is all ripped from the reltrans code, modified to return purely the 
!lensing factor.

!*****************************************************************************************************
subroutine getlens(a_spin,h,muobs,lens)
! Routine to calculate the lensing factor l=d\cos\delta/d\cos(i)
! and the source to observer time lag.
! Both calculations need us to know the delta value for the geodesic
! that ends up at angle i at infinity.
! INPUTS
! a_spin       Dimensionless spin parameter
! h            Height of on-axis, isotropically emitting source
! muobs        Cosine of inclination angle
!
! OUTPUTS
! lens         Lensing factor
! delt         Source to observer time lag 
  use blcoordinate
  implicit none
  double precision sins,mus,a_spin,h,scal
  double precision velocity(3)
  double precision muobs,drtbis,cosidel
  double precision cosdelta1,dcosdelta,lens
  double precision mudiff
  double precision par(3),x1,x2,xacc,mu2
  external mudiff
  
! Settings
  scal      = 1.d0   !Meaningless scaling factor
  mus       = 1.d0   !Position of source: mus=0 means on-axis
  sins      = 0.d0   !sin of same angle
  velocity  = 0.0D0  !3-velocity of source
  dcosdelta = 1d-2   !Step in cosdelta used for differentiation
  xacc      = 1d-6   !Accuracy of minimisation routine

! First calculate the cosdelta corresponding to the input muobs
      
  !Set limits for minimisation routine
  call getlimits(sins,mus,a_spin,h,velocity,muobs,x1,x2)
  !Call minimisation routine
  par(1)=a_spin
  par(2)=h
  par(3)=muobs
  cosdelta1 = drtbis(mudiff,x1,x2,xacc,par)
      
! Now calculate the lensing factor
      
  !Make cosdelta a little bit bigger and calculate the new cosi
  mu2 = cosidel(cosdelta1+dcosdelta,sins,mus,a_spin,h,velocity) 
  !Finally calculate the lensing factor
  lens = dcosdelta / ( muobs - mu2 )
  return
end subroutine getlens
!*****************************************************************************************************


!-----------------------------------------------------------------------
subroutine getlimits(sins,mus,a_spin,h,velocity,muobs,x1,x2)
! Minimisation routine will numerically calculate cosdelta for a given cosi.
! To do that, we need limits that bracket only one root. 
! This routine works out sensible limits
  use blcoordinate
  implicit none
  double precision sins,mus,a_spin,h,velocity(3),muobs,x1,x2
  double precision cosdelta0,mua,cosidel,cosi,cosdelta
  !The first limit is always cosdelta=-1 (corresponding to cosi=1)
  !Can't take cosdelta too large because this will also braket
  !the ghost images solutions
  !Tactic: extrapolate the initially straight line function from
  !cosi = 1, to some well-chosen cosi value. The cosdelta resulting
  !From this extrapolation is my second limit.
  cosdelta0 = -0.98d0
  mua = cosidel(cosdelta0,sins,mus,a_spin,h,velocity)
  !Take the straight line from (cosi=1,cosdelta=-1) to (cosi=mua,cosdelta=cosdelta0)
  !and extrapolate down to cosi=-0.5
  cosi = -0.5
  cosdelta = (cosi-1.d0)*(cosdelta0+1.d0)/(mua-1.d0) - 1.0
  cosdelta = min( cosdelta , -muobs )  !-muobs is the Newtonian limit 
  !Use for limits
  x1 = -1.d0
  x2 = cosdelta
  return
end subroutine getlimits
!-----------------------------------------------------------------------

      
!-----------------------------------------------------------------------
function cosidel(cosdelta,sins,mus,a_spin,h,velocity)
! Inputs:
! cosdelta,sins,mus,a_spin,h,velocity
        
! Calculates cosi when given cosdelta and parameters
!
!        
  use blcoordinate
  implicit none
  double precision cosdelta,sins,mus,a_spin,h,velocity(3),cosidel
  double precision pr,pp,pt,lambda,q,f1234(4),ptotal
  double precision scal,p,ra,mua,phya,timea,sigmaa
  scal = 1.d0                  !Meaningless scaling factor
  pr   = cosdelta              !cosdelta
  pp   = sqrt( 1.d0 - pr**2 )  !sindelta
  pt   = 0.d0
  !Convert to LNRF (locally non-rotating reference frame)
  call initialdirection(pr,pt,pp,sins,mus,a_spin,h,velocity,lambda,q,f1234)
  !Now calculate ptotal (value of p-coordinate at infinity)
  ptotal = p_total(f1234(1),lambda,q,sins,mus,a_spin,h,scal)
  p = 0.9999d0 * ptotal
  call YNOGK(p,f1234,lambda,q,sins,mus,a_spin,h,scal,&
           ra,mua,phya,timea,sigmaa)
  cosidel = mua
  return
end function cosidel
!-----------------------------------------------------------------------
      
!-----------------------------------------------------------------------
function mudiff(cosdelta,par)
    !Calculates muobs (cosine of distant inclination angle) when given
    !cos(delta) (cosine of angle between initial photon trajectory and -z)
    use blcoordinate
    implicit none
    double precision mudiff,cosdelta,par(3)
    double precision a_spin,h,muobs
    double precision scal,mus,sins
    double precision velocity(3),sindelta,pp,pr,pt,lambda,q,f1234(4)
    double precision ptotal,x,y,z,xprev,yprev,zprev,delx,dely,delz
    double precision ra,mua,phya,timea,sigmaa,p,cosdum
    a_spin = par(1)
    h      = par(2)
    muobs  = par(3)
    velocity = 0.0D0
    sindelta = sqrt( 1.d0 - cosdelta**2 )
    if ( sindelta .eq. 0.d0 )then
        cosdum = 1.d0
    else
        !Calculate 4-momentum in source rest frame tetrad
        pp= sindelta
        pr= cosdelta
        pt= 0.d0
        !Convert to LNRF (locally non-rotating reference frame)
        scal = 1.d0
        mus  = 1.d0
        sins = 0.d0
        call initialdirection(pr,pt,pp,sins,mus,a_spin,h,velocity,lambda,q,f1234)
        ptotal = p_total(f1234(1),lambda,q,sins,mus,a_spin,h,scal)
        p = 0.9998d0 * ptotal
        call YNOGK(p,f1234,lambda,q,sins,mus,a_spin,h,scal,&
                   ra,mua,phya,timea,sigmaa)
        xprev = sqrt(ra**2+a_spin**2)*sqrt(1.d0-mua**2)*cos(phya)
        yprev = sqrt(ra**2+a_spin**2)*sqrt(1.d0-mua**2)*sin(phya)
        zprev = ra*mua
        p = 0.9999d0 * ptotal
        call YNOGK(p,f1234,lambda,q,sins,mus,a_spin,h,scal,&
                   ra,mua,phya,timea,sigmaa)
        x = sqrt(ra**2+a_spin**2)*sqrt(1.d0-mua**2)*cos(phya)
        y = sqrt(ra**2+a_spin**2)*sqrt(1.d0-mua**2)*sin(phya)
        z = ra*mua
        delx = x - xprev
        dely = y - yprev
        delz = z - zprev
        cosdum = delz / sqrt( delx**2 + dely**2 + delz**2 )
    end if
    
    mudiff = cosdum - muobs
    return
end function mudiff

!-----------------------------------------------------------------------