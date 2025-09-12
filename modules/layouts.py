# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 4.2.1-0-g80c4cb6)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
import wx.grid

###########################################################################
## Class MainInterface
###########################################################################

class MainInterface ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Quantitative Endoscopy", pos = wx.DefaultPosition, size = wx.Size( 1600,1000 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
        self.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOWTEXT ) )
        self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )

        s_Main = wx.BoxSizer( wx.VERTICAL )

        s_StatusBar = wx.BoxSizer( wx.HORIZONTAL )

        self.b_open = wx.Button( self, wx.ID_ANY, u"Open...", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_open.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FILE_OPEN, wx.ART_HELP_BROWSER ) )
        s_StatusBar.Add( self.b_open, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_openExporter = wx.Button( self, wx.ID_ANY, u"Export...", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_openExporter.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FILE_SAVE, wx.ART_HELP_BROWSER ) )
        self.b_openExporter.Enable( False )

        s_StatusBar.Add( self.b_openExporter, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_openViewer = wx.Button( self, wx.ID_ANY, u"View Measurements...", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_openViewer.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_LIST_VIEW, wx.ART_TOOLBAR ) )
        self.b_openViewer.Enable( False )

        s_StatusBar.Add( self.b_openViewer, 0, wx.ALL|wx.EXPAND, 5 )


        s_StatusBar.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.b_help = wx.Button( self, wx.ID_ANY, u"Help", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_help.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_QUESTION, wx.ART_BUTTON ) )
        s_StatusBar.Add( self.b_help, 0, wx.ALL|wx.EXPAND, 5 )


        s_Main.Add( s_StatusBar, 0, wx.EXPAND, 5 )

        s_mainarea = wx.FlexGridSizer( 0, 2, 5, 5 )
        s_mainarea.AddGrowableCol( 0 )
        s_mainarea.AddGrowableCol( 1 )
        s_mainarea.AddGrowableRow( 0 )
        s_mainarea.SetFlexibleDirection( wx.HORIZONTAL )
        s_mainarea.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

        self.i_Image = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 10,10 ), wx.FULL_REPAINT_ON_RESIZE )
        self.i_Image.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BACKGROUND ) )

        s_mainarea.Add( self.i_Image, 1, wx.ALIGN_CENTER_HORIZONTAL|wx.SHAPED, 0 )

        self.vispypanel = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,-1 ), wx.TAB_TRAVERSAL )
        self.vispypanel.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BACKGROUND ) )

        s_mainarea.Add( self.vispypanel, 1, wx.EXPAND, 5 )

        s_leftcontrols = wx.BoxSizer( wx.VERTICAL )

        s_videotools = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Imaging Tools" ), wx.VERTICAL )

        s_transport = wx.BoxSizer( wx.HORIZONTAL )

        self.b_framePrevious = wx.Button( s_videotools.GetStaticBox(), wx.ID_ANY, u"<", wx.DefaultPosition, wx.Size( 60,-1 ), wx.BU_NOTEXT )

        self.b_framePrevious.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_GO_BACK, wx.ART_HELP_BROWSER ) )
        s_transport.Add( self.b_framePrevious, 0, 0, 5 )

        self.slider_time = wx.Slider( s_videotools.GetStaticBox(), wx.ID_ANY, 0, 0, 1, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL|wx.SL_VALUE_LABEL )
        s_transport.Add( self.slider_time, 1, wx.EXPAND, 10 )

        self.b_FrameNext = wx.Button( s_videotools.GetStaticBox(), wx.ID_ANY, u">", wx.DefaultPosition, wx.Size( 60,-1 ), wx.BU_NOTEXT )

        self.b_FrameNext.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_GO_FORWARD, wx.ART_HELP_BROWSER ) )
        s_transport.Add( self.b_FrameNext, 0, 0, 5 )


        s_videotools.Add( s_transport, 0, wx.ALL|wx.EXPAND, 5 )

        s_grid1 = wx.GridSizer( 0, 3, 0, 0 )

        self.b_playVideo = wx.ToggleButton( s_videotools.GetStaticBox(), wx.ID_ANY, u"Play/Pause Video", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_playVideo.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_GO_FORWARD, wx.ART_MENU ) )
        self.b_playVideo.SetBitmapPressed( wx.ArtProvider.GetBitmap( wx.ART_DELETE, wx.ART_MENU ) )
        self.b_playVideo.Enable( False )

        s_grid1.Add( self.b_playVideo, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_jumpFrame = wx.Button( s_videotools.GetStaticBox(), wx.ID_ANY, u"Jump To...", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_jumpFrame.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_GOTO_LAST, wx.ART_HELP_BROWSER ) )
        self.b_jumpFrame.Enable( False )

        s_grid1.Add( self.b_jumpFrame, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_showSettings = wx.Button( s_videotools.GetStaticBox(), wx.ID_ANY, u"Video Settings...", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_showSettings.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FIND_AND_REPLACE, wx.ART_HELP_BROWSER ) )
        self.b_showSettings.Enable( False )

        s_grid1.Add( self.b_showSettings, 0, wx.ALL|wx.EXPAND, 5 )


        s_videotools.Add( s_grid1, 1, wx.EXPAND, 5 )


        s_leftcontrols.Add( s_videotools, 0, wx.EXPAND, 5 )

        s_measurements = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Message Log" ), wx.VERTICAL )

        self.t_statusText = wx.TextCtrl( s_measurements.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE|wx.TE_READONLY|wx.TE_WORDWRAP|wx.BORDER_NONE )
        s_measurements.Add( self.t_statusText, 1, wx.ALL|wx.EXPAND, 5 )


        s_leftcontrols.Add( s_measurements, 1, wx.EXPAND, 5 )


        s_mainarea.Add( s_leftcontrols, 1, wx.EXPAND, 5 )

        s_rightcontrols = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Measurement Tools" ), wx.VERTICAL )

        s_grid2 = wx.FlexGridSizer( 0, 2, 5, 10 )
        s_grid2.AddGrowableCol( 1 )
        s_grid2.SetFlexibleDirection( wx.BOTH )
        s_grid2.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_NONE )

        self.t_toolchoice = wx.StaticText( s_rightcontrols.GetStaticBox(), wx.ID_ANY, u"Slice Distance:", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.t_toolchoice.Wrap( -1 )

        s_grid2.Add( self.t_toolchoice, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT, 5 )

        self.slider_distance = wx.Slider( s_rightcontrols.GetStaticBox(), wx.ID_ANY, 1000, 0, 5000, wx.DefaultPosition, wx.DefaultSize, wx.SL_BOTTOM|wx.SL_VALUE_LABEL )
        self.slider_distance.SetFont( wx.Font( 8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )

        s_grid2.Add( self.slider_distance, 0, wx.EXPAND, 5 )

        self.t_rotateX = wx.StaticText( s_rightcontrols.GetStaticBox(), wx.ID_ANY, u"Rotate Slice (X):", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.t_rotateX.Wrap( -1 )

        s_grid2.Add( self.t_rotateX, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT, 5 )

        self.slider_planeX = wx.Slider( s_rightcontrols.GetStaticBox(), wx.ID_ANY, 0, -45, 45, wx.DefaultPosition, wx.DefaultSize, wx.SL_VALUE_LABEL )
        self.slider_planeX.SetFont( wx.Font( 8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )

        s_grid2.Add( self.slider_planeX, 0, wx.EXPAND, 5 )

        self.t_rotateY = wx.StaticText( s_rightcontrols.GetStaticBox(), wx.ID_ANY, u"Rotate Slice (Y):", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.t_rotateY.Wrap( -1 )

        s_grid2.Add( self.t_rotateY, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT, 5 )

        self.slider_planeY = wx.Slider( s_rightcontrols.GetStaticBox(), wx.ID_ANY, 0, -45, 45, wx.DefaultPosition, wx.DefaultSize, wx.SL_VALUE_LABEL )
        self.slider_planeY.SetFont( wx.Font( 8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )

        s_grid2.Add( self.slider_planeY, 0, wx.EXPAND, 5 )


        s_rightcontrols.Add( s_grid2, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, 5 )

        s_grid3 = wx.GridSizer( 0, 3, 0, 0 )

        self.b_recordMeasurement = wx.Button( s_rightcontrols.GetStaticBox(), wx.ID_ANY, u"Record Measurement", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_recordMeasurement.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_PLUS, wx.ART_HELP_BROWSER ) )
        self.b_recordMeasurement.Enable( False )

        s_grid3.Add( self.b_recordMeasurement, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_showmap = wx.Button( s_rightcontrols.GetStaticBox(), wx.ID_ANY, u"Show Depth Map...", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_showmap.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FIND, wx.ART_HELP_BROWSER ) )
        self.b_showmap.Enable( False )

        s_grid3.Add( self.b_showmap, 0, wx.ALL|wx.EXPAND, 5 )

        b_vischoiceChoices = [ u"Show All", u"Hide Slicing Plane", u"Show Selection Only" ]
        self.b_vischoice = wx.Choice( s_rightcontrols.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, b_vischoiceChoices, 0 )
        self.b_vischoice.SetSelection( 0 )
        s_grid3.Add( self.b_vischoice, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_resetPlane = wx.Button( s_rightcontrols.GetStaticBox(), wx.ID_ANY, u"Reset Tools", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_resetPlane.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_MISSING_IMAGE, wx.ART_HELP_BROWSER ) )
        s_grid3.Add( self.b_resetPlane, 0, wx.ALL|wx.EXPAND, 5 )

        self.b_resetCamera = wx.Button( s_rightcontrols.GetStaticBox(), wx.ID_ANY, u"Reset Camera", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_resetCamera.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_GO_HOME, wx.ART_HELP_BROWSER ) )
        s_grid3.Add( self.b_resetCamera, 0, wx.ALL|wx.EXPAND, 5 )

        b_colormapChoices = [ u"Monochrome", u"Inferno", u"Bone", u"Rainbow" ]
        self.b_colormap = wx.Choice( s_rightcontrols.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, b_colormapChoices, 0 )
        self.b_colormap.SetSelection( 0 )
        s_grid3.Add( self.b_colormap, 0, wx.ALL|wx.EXPAND, 5 )


        s_rightcontrols.Add( s_grid3, 0, wx.EXPAND, 5 )


        s_mainarea.Add( s_rightcontrols, 1, wx.EXPAND, 5 )


        s_Main.Add( s_mainarea, 1, wx.EXPAND, 5 )


        self.SetSizer( s_Main )
        self.Layout()

        self.Centre( wx.BOTH )

        # Connect Events
        self.Bind( wx.EVT_CLOSE, self.saveAndQuit )
        self.Bind( wx.EVT_MAXIMIZE, self.setSize )
        self.Bind( wx.EVT_MOVE_END, self.setSize )
        self.Bind( wx.EVT_SHOW, self.setSize )
        self.b_open.Bind( wx.EVT_BUTTON, self.openVideo )
        self.b_openExporter.Bind( wx.EVT_BUTTON, self.openExporter )
        self.b_openViewer.Bind( wx.EVT_BUTTON, self.openViewer )
        self.b_help.Bind( wx.EVT_BUTTON, self.openHelp )
        self.i_Image.Bind( wx.EVT_LEFT_DOWN, self.pickPoint )
        self.i_Image.Bind( wx.EVT_RIGHT_DOWN, self.clearMeasurements )
        self.b_framePrevious.Bind( wx.EVT_BUTTON, self.framePrevious )
        self.slider_time.Bind( wx.EVT_SCROLL_CHANGED, self.scrubDone )
        self.slider_time.Bind( wx.EVT_SLIDER, self.scrub )
        self.b_FrameNext.Bind( wx.EVT_BUTTON, self.frameNext )
        self.b_playVideo.Bind( wx.EVT_TOGGLEBUTTON, self.playVideo )
        self.b_jumpFrame.Bind( wx.EVT_BUTTON, self.jumpFrame )
        self.b_showSettings.Bind( wx.EVT_BUTTON, self.showSettings )
        self.t_statusText.Bind( wx.EVT_LEFT_DOWN, self.copyText )
        self.slider_distance.Bind( wx.EVT_SCROLL_CHANGED, self.slicerDone )
        self.slider_distance.Bind( wx.EVT_SLIDER, self.slicerChange )
        self.slider_planeX.Bind( wx.EVT_SCROLL, self.threshChange )
        self.slider_planeX.Bind( wx.EVT_SCROLL_CHANGED, self.slicerDone )
        self.slider_planeX.Bind( wx.EVT_SLIDER, self.slicerChange )
        self.slider_planeY.Bind( wx.EVT_SCROLL_CHANGED, self.slicerDone )
        self.slider_planeY.Bind( wx.EVT_SLIDER, self.slicerChange )
        self.b_recordMeasurement.Bind( wx.EVT_BUTTON, self.recordMeasurement )
        self.b_showmap.Bind( wx.EVT_BUTTON, self.showmap )
        self.b_vischoice.Bind( wx.EVT_CHOICE, self.changeVisibility )
        self.b_resetPlane.Bind( wx.EVT_BUTTON, self.resetPlane )
        self.b_resetCamera.Bind( wx.EVT_BUTTON, self.resetCamera )
        self.b_colormap.Bind( wx.EVT_CHOICE, self.setColors )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def saveAndQuit( self, event ):
        pass

    def setSize( self, event ):
        pass



    def openVideo( self, event ):
        pass

    def openExporter( self, event ):
        pass

    def openViewer( self, event ):
        pass

    def openHelp( self, event ):
        pass

    def pickPoint( self, event ):
        pass

    def clearMeasurements( self, event ):
        pass

    def framePrevious( self, event ):
        pass

    def scrubDone( self, event ):
        pass

    def scrub( self, event ):
        pass

    def frameNext( self, event ):
        pass

    def playVideo( self, event ):
        pass

    def jumpFrame( self, event ):
        pass

    def showSettings( self, event ):
        pass

    def copyText( self, event ):
        pass

    def slicerDone( self, event ):
        pass

    def slicerChange( self, event ):
        pass

    def threshChange( self, event ):
        pass





    def recordMeasurement( self, event ):
        pass

    def showmap( self, event ):
        pass

    def changeVisibility( self, event ):
        pass

    def resetPlane( self, event ):
        pass

    def resetCamera( self, event ):
        pass

    def setColors( self, event ):
        pass


###########################################################################
## Class VideoSettings
###########################################################################

class VideoSettings ( wx.Dialog ):

    def __init__( self, parent ):
        wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = u"Video Settings Editor", pos = wx.DefaultPosition, size = wx.Size( 1200,750 ), style = wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER )

        self.SetSizeHints( wx.Size( -1,-1 ), wx.DefaultSize )

        s_Trimmer = wx.BoxSizer( wx.VERTICAL )

        fgSizer1 = wx.FlexGridSizer( 3, 2, 0, 0 )
        fgSizer1.AddGrowableCol( 0 )
        fgSizer1.AddGrowableCol( 1 )
        fgSizer1.AddGrowableRow( 1 )
        fgSizer1.SetFlexibleDirection( wx.BOTH )
        fgSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_ALL )

        self.t_Start = wx.StaticText( self, wx.ID_ANY, u"Start Frame", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL )
        self.t_Start.Wrap( -1 )

        fgSizer1.Add( self.t_Start, 1, wx.ALIGN_CENTER, 5 )

        self.t_End = wx.StaticText( self, wx.ID_ANY, u"End Frame", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL )
        self.t_End.Wrap( -1 )

        fgSizer1.Add( self.t_End, 1, wx.ALIGN_CENTER, 5 )

        self.i_TrimStart = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 500,500 ), 0 )
        self.i_TrimStart.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNTEXT ) )

        fgSizer1.Add( self.i_TrimStart, 1, wx.ALIGN_CENTER|wx.SHAPED, 5 )

        self.i_TrimEnd = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 500,500 ), 0 )
        self.i_TrimEnd.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNTEXT ) )

        fgSizer1.Add( self.i_TrimEnd, 1, wx.ALIGN_CENTER|wx.SHAPED, 5 )

        self.b_TrimStart = wx.Slider( self, wx.ID_ANY, 0, 0, 100, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL|wx.SL_VALUE_LABEL )
        fgSizer1.Add( self.b_TrimStart, 1, wx.EXPAND, 5 )

        self.b_TrimEnd = wx.Slider( self, wx.ID_ANY, 100, 0, 100, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL|wx.SL_VALUE_LABEL )
        fgSizer1.Add( self.b_TrimEnd, 1, wx.EXPAND, 5 )


        s_Trimmer.Add( fgSizer1, 0, wx.ALL|wx.EXPAND, 5 )

        self.t_trimTitleLine2 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        s_Trimmer.Add( self.t_trimTitleLine2, 0, wx.EXPAND |wx.ALL, 5 )

        bSizer14 = wx.BoxSizer( wx.HORIZONTAL )


        bSizer14.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.t_crop = wx.StaticText( self, wx.ID_ANY, u"Horizontal Offset:", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.t_crop.Wrap( -1 )

        bSizer14.Add( self.t_crop, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.b_cropOffset = wx.SpinCtrl( self, wx.ID_ANY, u"0", wx.Point( -1,-1 ), wx.Size( 150,-1 ), wx.ALIGN_CENTER_HORIZONTAL|wx.SP_ARROW_KEYS, -300, 300, 0 )
        bSizer14.Add( self.b_cropOffset, 0, wx.ALIGN_CENTER_VERTICAL, 5 )


        bSizer14.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.t_zoom = wx.StaticText( self, wx.ID_ANY, u"Zoom:", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.t_zoom.Wrap( -1 )

        bSizer14.Add( self.t_zoom, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.b_zoom = wx.SpinCtrl( self, wx.ID_ANY, u"0", wx.Point( -1,-1 ), wx.Size( 150,-1 ), wx.ALIGN_CENTER_HORIZONTAL|wx.SP_ARROW_KEYS, 0, 250, 0 )
        bSizer14.Add( self.b_zoom, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        bSizer14.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.t_speed = wx.StaticText( self, wx.ID_ANY, u"Playback Speed:", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_RIGHT )
        self.t_speed.Wrap( -1 )

        bSizer14.Add( self.t_speed, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.b_speed = wx.SpinCtrlDouble( self, wx.ID_ANY, u"1", wx.DefaultPosition, wx.Size( 150,-1 ), wx.ALIGN_CENTER_HORIZONTAL|wx.SP_ARROW_KEYS, 0.1, 4, 1, 0.2 )
        self.b_speed.SetDigits( 2 )
        bSizer14.Add( self.b_speed, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        bSizer14.Add( ( 0, 0), 1, wx.EXPAND, 5 )


        s_Trimmer.Add( bSizer14, 1, wx.ALL|wx.EXPAND, 5 )

        self.t_trimTitleLine1 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        s_Trimmer.Add( self.t_trimTitleLine1, 0, wx.EXPAND |wx.ALL, 5 )

        bSizer10 = wx.BoxSizer( wx.HORIZONTAL )

        self.t_duration = wx.StaticText( self, wx.ID_ANY, u"Clip Duration:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.t_duration.Wrap( -1 )

        bSizer10.Add( self.t_duration, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        bSizer10.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.b_TrimDone = wx.Button( self, wx.ID_ANY, u"Apply Changes", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.b_TrimDone.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FILE_SAVE, wx.ART_HELP_BROWSER ) )
        self.b_TrimDone.SetMinSize( wx.Size( 200,-1 ) )

        bSizer10.Add( self.b_TrimDone, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )


        s_Trimmer.Add( bSizer10, 1, wx.ALL|wx.EXPAND, 5 )


        self.SetSizer( s_Trimmer )
        self.Layout()

        self.Centre( wx.BOTH )

        # Connect Events
        self.Bind( wx.EVT_CLOSE, self.doneTrimming )
        self.b_TrimStart.Bind( wx.EVT_SLIDER, self.updateTrim )
        self.b_TrimEnd.Bind( wx.EVT_SLIDER, self.updateTrim )
        self.b_cropOffset.Bind( wx.EVT_SPINCTRL, self.updateCrop )
        self.b_cropOffset.Bind( wx.EVT_TEXT, self.updateCrop )
        self.b_cropOffset.Bind( wx.EVT_TEXT_ENTER, self.updateCrop )
        self.b_zoom.Bind( wx.EVT_SPINCTRL, self.updateCrop )
        self.b_zoom.Bind( wx.EVT_TEXT, self.updateCrop )
        self.b_zoom.Bind( wx.EVT_TEXT_ENTER, self.updateCrop )
        self.b_TrimDone.Bind( wx.EVT_BUTTON, self.doneTrimming )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def doneTrimming( self, event ):
        pass

    def updateTrim( self, event ):
        pass


    def updateCrop( self, event ):
        pass








###########################################################################
## Class Measurements
###########################################################################

class Measurements ( wx.Dialog ):

    def __init__( self, parent ):
        wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = u"Measurement Table Viewer", pos = wx.DefaultPosition, size = wx.Size( 1100,800 ), style = wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

        s_exportMain = wx.BoxSizer( wx.VERTICAL )

        self.grid = wx.grid.Grid( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0 )

        # Grid
        self.grid.CreateGrid( 1, 8 )
        self.grid.EnableEditing( False )
        self.grid.EnableGridLines( True )
        self.grid.EnableDragGridSize( False )
        self.grid.SetMargins( 20, 20 )

        # Columns
        self.grid.AutoSizeColumns()
        self.grid.EnableDragColMove( False )
        self.grid.EnableDragColSize( True )
        self.grid.SetColLabelValue( 0, u"i" )
        self.grid.SetColLabelValue( 1, u"Frame No." )
        self.grid.SetColLabelValue( 2, u"Timecode" )
        self.grid.SetColLabelValue( 3, u"Pointcloud Range" )
        self.grid.SetColLabelValue( 4, u"Plane Distance" )
        self.grid.SetColLabelValue( 5, u"Plane Rotation" )
        self.grid.SetColLabelValue( 6, u"Matched Points" )
        self.grid.SetColLabelValue( 7, u"Cross Section Area" )
        self.grid.SetColLabelSize( wx.grid.GRID_AUTOSIZE )
        self.grid.SetColLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )

        # Rows
        self.grid.SetRowSize( 0, 0 )
        self.grid.EnableDragRowSize( False )
        self.grid.SetRowLabelSize( 0 )
        self.grid.SetRowLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )

        # Label Appearance

        # Cell Defaults
        self.grid.SetDefaultCellFont( wx.Font( 9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )
        self.grid.SetDefaultCellAlignment( wx.ALIGN_LEFT, wx.ALIGN_CENTER )
        s_exportMain.Add( self.grid, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, 5 )

        self.b_close = wx.Button( self, wx.ID_ANY, u"OK", wx.DefaultPosition, wx.DefaultSize, 0 )
        s_exportMain.Add( self.b_close, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )


        self.SetSizer( s_exportMain )
        self.Layout()

        self.Centre( wx.BOTH )

        # Connect Events
        self.b_close.Bind( wx.EVT_BUTTON, self.closeTable )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def closeTable( self, event ):
        pass


###########################################################################
## Class SaveWizard
###########################################################################

class SaveWizard ( wx.Dialog ):

    def __init__( self, parent ):
        wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = u"Batch Exporter", pos = wx.DefaultPosition, size = wx.Size( 652,505 ), style = wx.DEFAULT_DIALOG_STYLE )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

        saveSizer = wx.BoxSizer( wx.VERTICAL )

        self.t_saveTitle = wx.StaticText( self, wx.ID_ANY, u"Select Formats to Export", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL )
        self.t_saveTitle.Wrap( -1 )

        self.t_saveTitle.SetFont( wx.Font( 14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, "Arial" ) )

        saveSizer.Add( self.t_saveTitle, 0, wx.ALL|wx.EXPAND, 5 )

        saveSizer2 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Endoscopy Image (video frame)" ), wx.HORIZONTAL )

        self.cb_image = wx.CheckBox( saveSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        saveSizer2.Add( self.cb_image, 0, wx.ALL, 5 )

        self.t_imageName = wx.TextCtrl( saveSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        saveSizer2.Add( self.t_imageName, 1, 0, 5 )

        c_imageFmtChoices = [ u".jpg", u".png" ]
        self.c_imageFmt = wx.Choice( saveSizer2.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, c_imageFmtChoices, 0 )
        self.c_imageFmt.SetSelection( 0 )
        saveSizer2.Add( self.c_imageFmt, 0, 0, 5 )


        saveSizer.Add( saveSizer2, 0, wx.ALL|wx.EXPAND, 5 )

        saveSizer3 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Depth Map (image or data)" ), wx.HORIZONTAL )

        self.cb_depthmap = wx.CheckBox( saveSizer3.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        saveSizer3.Add( self.cb_depthmap, 0, wx.ALL, 5 )

        self.t_depthmapName = wx.TextCtrl( saveSizer3.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        saveSizer3.Add( self.t_depthmapName, 1, 0, 5 )

        c_depthmapFmtChoices = [ u".png", u".jpg", u".npy", u".csv" ]
        self.c_depthmapFmt = wx.Choice( saveSizer3.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, c_depthmapFmtChoices, 0 )
        self.c_depthmapFmt.SetSelection( 0 )
        saveSizer3.Add( self.c_depthmapFmt, 0, 0, 5 )


        saveSizer.Add( saveSizer3, 0, wx.ALL|wx.EXPAND, 5 )

        saveSizer4 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Point Cloud Data" ), wx.HORIZONTAL )

        self.cb_pointcloud = wx.CheckBox( saveSizer4.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        saveSizer4.Add( self.cb_pointcloud, 0, wx.ALL, 5 )

        self.t_pointcloudName = wx.TextCtrl( saveSizer4.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        saveSizer4.Add( self.t_pointcloudName, 1, 0, 5 )

        c_pointcloudFmtChoices = [ u".npy", u".xyz" ]
        self.c_pointcloudFmt = wx.Choice( saveSizer4.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, c_pointcloudFmtChoices, 0 )
        self.c_pointcloudFmt.SetSelection( 0 )
        saveSizer4.Add( self.c_pointcloudFmt, 0, 0, 5 )


        saveSizer.Add( saveSizer4, 0, wx.ALL|wx.EXPAND, 5 )

        saveSizer5 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Measurement Table" ), wx.HORIZONTAL )

        self.cb_measurement = wx.CheckBox( saveSizer5.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        saveSizer5.Add( self.cb_measurement, 0, wx.ALL, 5 )

        self.t_measurementName = wx.TextCtrl( saveSizer5.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        saveSizer5.Add( self.t_measurementName, 1, 0, 5 )

        c_measurementFmtChoices = [ u".xlsx", u".csv" ]
        self.c_measurementFmt = wx.Choice( saveSizer5.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, c_measurementFmtChoices, 0 )
        self.c_measurementFmt.SetSelection( 0 )
        saveSizer5.Add( self.c_measurementFmt, 0, 0, 5 )


        saveSizer.Add( saveSizer5, 0, wx.ALL|wx.EXPAND, 5 )


        saveSizer.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        saveSizer6 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Output Directory" ), wx.HORIZONTAL )


        saveSizer6.Add( ( 28, 0), 0, 0, 5 )

        self.outputDirectory = wx.DirPickerCtrl( saveSizer6.GetStaticBox(), wx.ID_ANY, wx.EmptyString, u"Select a folder", wx.DefaultPosition, wx.DefaultSize, wx.DIRP_DEFAULT_STYLE )
        saveSizer6.Add( self.outputDirectory, 1, wx.ALL|wx.EXPAND, 5 )


        saveSizer.Add( saveSizer6, 1, wx.ALL|wx.EXPAND, 5 )

        self.m_staticline5 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        saveSizer.Add( self.m_staticline5, 0, wx.EXPAND |wx.ALL, 5 )

        self.b_dosave = wx.Button( self, wx.ID_ANY, u"Save", wx.DefaultPosition, wx.Size( 200,-1 ), 0 )

        self.b_dosave.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FILE_SAVE, wx.ART_TOOLBAR ) )
        saveSizer.Add( self.b_dosave, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )


        self.SetSizer( saveSizer )
        self.Layout()

        self.Centre( wx.BOTH )

        # Connect Events
        self.b_dosave.Bind( wx.EVT_BUTTON, self.doSave )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def doSave( self, event ):
        pass


###########################################################################
## Class DepthmapViewer
###########################################################################

class DepthmapViewer ( wx.Dialog ):

    def __init__( self, parent ):
        wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = u"Depthmap Viewer", pos = wx.DefaultPosition, size = wx.DefaultSize, style = wx.CAPTION|wx.CLOSE_BOX|wx.SYSTEM_MENU )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

        dm_sizer1 = wx.BoxSizer( wx.VERTICAL )

        self.i_depthmap = wx.StaticBitmap( self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.Size( 720,720 ), 0 )
        dm_sizer1.Add( self.i_depthmap, 0, wx.ALL|wx.SHAPED, 5 )

        dm_sizer2 = wx.BoxSizer( wx.HORIZONTAL )

        self.b_colormap = wx.CheckBox( self, wx.ID_ANY, u"Remap colors", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.b_colormap.SetValue(True)
        dm_sizer2.Add( self.b_colormap, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5 )

        self.t_stats = wx.StaticText( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL )
        self.t_stats.Wrap( -1 )

        dm_sizer2.Add( self.t_stats, 1, wx.ALIGN_CENTER_VERTICAL, 5 )

        self.m_closeViewer = wx.Button( self, wx.ID_ANY, u"OK", wx.DefaultPosition, wx.DefaultSize, 0 )

        self.m_closeViewer.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_TICK_MARK, wx.ART_TOOLBAR ) )
        dm_sizer2.Add( self.m_closeViewer, 0, wx.ALIGN_BOTTOM|wx.ALL, 5 )


        dm_sizer1.Add( dm_sizer2, 1, wx.EXPAND, 5 )


        self.SetSizer( dm_sizer1 )
        self.Layout()
        dm_sizer1.Fit( self )

        self.Centre( wx.BOTH )

        # Connect Events
        self.b_colormap.Bind( wx.EVT_CHECKBOX, self.toggleColors )
        self.m_closeViewer.Bind( wx.EVT_BUTTON, self.closeViewer )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def toggleColors( self, event ):
        pass

    def closeViewer( self, event ):
        pass


