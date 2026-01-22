'use client';

interface ScanFrameProps {
  darkMode: boolean;
}

export const ScanFrame: React.FC<ScanFrameProps> = ({ darkMode }) => {
  return (
    <div className="relative w-full mx-auto mb-6">
      {/* Viewfinder container with aspect ratio */}
      <div className="relative w-full max-w-md mx-auto">
        {/* Aspect ratio wrapper - 3:4 for portrait camera view */}
        <div className="relative w-full pb-[133.33%] lg:pb-[100%]">
          {/* Actual viewfinder content */}
          <div 
            className="absolute inset-0 rounded-2xl lg:rounded-3xl overflow-hidden shadow-2xl flex items-center justify-center"
            style={{
              background: 'linear-gradient(180deg, #101828 0%, #1E2939 50%, #101828 100%)'
            }}
          >
            
            {/* Subtle grid overlay for alignment */}
            <div className="absolute inset-0 pointer-events-none opacity-20">
              <div className="absolute top-1/2 left-0 w-full h-px bg-neutral-400"></div>
              <div className="absolute top-0 left-1/2 w-px h-full bg-neutral-400"></div>
            </div>

            {/* Corner brackets - more refined and subtle */}
            <div className="absolute inset-0 pointer-events-none p-6 lg:p-8">
              {/* Top left */}
              <div className="absolute top-6 left-6 lg:top-8 lg:left-8">
                <div className={`w-12 h-12 lg:w-16 lg:h-16 border-t-[3px] border-l-[3px] rounded-tl-md ${
                  darkMode ? 'border-neutral-500/80' : 'border-neutral-400/80'
                }`}></div>
              </div>
              {/* Top right */}
              <div className="absolute top-6 right-6 lg:top-8 lg:right-8">
                <div className={`w-12 h-12 lg:w-16 lg:h-16 border-t-[3px] border-r-[3px] rounded-tr-md ${
                  darkMode ? 'border-neutral-500/80' : 'border-neutral-400/80'
                }`}></div>
              </div>
              {/* Bottom left */}
              <div className="absolute bottom-6 left-6 lg:bottom-8 lg:left-8">
                <div className={`w-12 h-12 lg:w-16 lg:h-16 border-b-[3px] border-l-[3px] rounded-bl-md ${
                  darkMode ? 'border-neutral-500/80' : 'border-neutral-400/80'
                }`}></div>
              </div>
              {/* Bottom right */}
              <div className="absolute bottom-6 right-6 lg:bottom-8 lg:right-8">
                <div className={`w-12 h-12 lg:w-16 lg:h-16 border-b-[3px] border-r-[3px] rounded-br-md ${
                  darkMode ? 'border-neutral-500/80' : 'border-neutral-400/80'
                }`}></div>
              </div>
            </div>

            {/* Center instruction text */}
            <div className="relative z-10 px-8 text-center">
              <p className={`text-base lg:text-lg font-medium tracking-wide ${
                darkMode ? 'text-neutral-300' : 'text-neutral-300'
              } drop-shadow-lg`}>
                Position plant in frame
              </p>
              <p className={`mt-2 text-xs lg:text-sm ${
                darkMode ? 'text-neutral-400' : 'text-neutral-400'
              } drop-shadow`}>
                Center your plant for best results
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
